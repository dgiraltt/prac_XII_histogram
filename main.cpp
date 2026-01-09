#include <iostream>
#include <oneapi/tbb.h>
#include <random>
#include <vector>


// Global constants for configuration
const int NUM_BINS = 10;	 	// Number of histogram bins
const int DATA_SIZE = 10000000;	// Size of the input data vector


/**
 * Prints detailed bin information for debugging purposes.
 * For each bin, displays the bin index, value range, and frequency count.
 * Useful for verifying correct bin assignments and 
 *  understanding data distribution.
 * 
 * @param bins Vector containing frequency counts for each bin.
 * @param min_val Minimum value in the data range.
 * @param max_val Maximum value in the data range.
 */
void printBinDetails(const std::vector<int> &bins, 
					 int min_val, 
					 int max_val)
{
	std::cout << "\nHistogram Data by Bins:" << std::endl;
	double range = (double)(max_val - min_val + 1);
	double bin_width = range / bins.size();
	for (size_t i = 0; i < bins.size(); i++)
	{
		int start = min_val + (int)(i * bin_width);
		int end = min_val + (int)((i + 1) * bin_width);
		std::cout << "  Bin " << i << " [" << start << ", " << end << "): " 
				  << bins[i] << std::endl;
	}
}


/**
 * Computes a sequential histogram with accumulative (prefix sum) values.
 * First counts the frequency of values in each bin, then performs a
 * sequential accumulation to convert frequencies into cumulative counts.
 * Supports both positive and negative values.
 * 
 * @param data Input vector containing integer values to be binned.
 * @param num_bins Number of bins to divide the value range into.
 * @param min_val Minimum value in the expected data range.
 * @param max_val Maximum value in the expected data range.
 * 
 * @return Vector of integers containing cumulative histogram values for each bin.
 */
std::vector<int> sequentialHistogram(const std::vector<int> &data, 
									 int num_bins, 
									 int min_val, 
									 int max_val)
{
	std::vector<int> bins(num_bins, 0);
	double range = (double)(max_val - min_val + 1);
	double bin_width = range / num_bins;

	// Iterative binning operation
	for (int val : data)
	{
		int bin_idx = std::min((int)((val - min_val) / bin_width), num_bins - 1);
		bins[bin_idx]++;
	}

	printBinDetails(bins, min_val, max_val);

	// Bin accumulation
	for (size_t i = 1; i < bins.size(); i++)
	{
		bins[i] += bins[i - 1];
	}
	return bins;
}


/**
 * Computes a fully parallel histogram using TBB parallel_reduce for frequency
 *  counting and TBB parallel_scan for the accumulation step.
 * This implementation parallelizes both the binning and prefix sum operations, 
 * though for small bin counts, the overhead may outweigh the benefits.
 * Supports both positive and negative values.
 * 
 * @param data Input vector containing integer values to be binned.
 * @param num_bins Number of bins to divide the value range into.
 * @param min_val Minimum value in the expected data range.
 * @param max_val Maximum value in the expected data range.
 * 
 * @return Vector of integers containing cumulative histogram values for each bin.
 */
std::vector<int> parallelHistogram(const std::vector<int> &data, 
								   int num_bins, 
								   int min_val, 
								   int max_val)
{
	// Compute Frequencies using Parallel Reduce
	double range = (double)(max_val - min_val + 1);
	double bin_width = range / num_bins;
	std::vector<int> bins = oneapi::tbb::parallel_reduce(
		oneapi::tbb::blocked_range<size_t>(0, data.size()),
		std::vector<int>(num_bins, 0),
		
		// Binning classification
		[&](const oneapi::tbb::blocked_range<size_t> &r, std::vector<int> local_bins) -> std::vector<int>
		{
			for (size_t i = r.begin(); i != r.end(); i++)
			{
				int val = data[i];
				int bin_idx = std::min((int)((val - min_val) / bin_width), num_bins - 1);
				local_bins[bin_idx]++;
			}
			return local_bins;
		},
		
		// Bins combination
		[](std::vector<int> a, const std::vector<int> &b) -> std::vector<int>
		{
			for (size_t i = 0; i < a.size(); i++)
			{
				a[i] += b[i];
			}
			return a;
		});

	// Parallel Scan for the Accumulation Step
	oneapi::tbb::parallel_scan(
		oneapi::tbb::blocked_range<size_t>(0, bins.size()),
		0,

		// Scan operation
		[&bins](const oneapi::tbb::blocked_range<size_t> &r, int sum, bool is_final_scan) -> int
		{
			int temp = sum;
			for (size_t i = r.begin(); i != r.end(); i++)
			{
				temp = temp + bins[i];
				if (is_final_scan)
				{
					bins[i] = temp;
				}
			}
			return temp;
		},

		// Combine operation
		[](int a, int b) -> int
		{ return a + b; });

	return bins;
}


/**
 * Randomly generates a vector with integers.
 *
 * @param size Nnumber of elements of the vector.
 * 
 * @return std::vector<int> containing the random integers.
 */
std::vector<int> random_vector(int size)
{
	const int MIN_VALUE = -500;  	// Minimum value (supports negative numbers)
	const int MAX_VALUE = 1000;  	// Maximum value
	
	std::mt19937 gen(13);
	std::uniform_int_distribution<> dis(MIN_VALUE, MAX_VALUE);
	
    std::vector<int> V(size);
    for (int &i : V)
    {
        i = std::min(MAX_VALUE, dis(gen));
    }

    return V;
}


/**
 * Finds either the minimum or maximum value in a vector using parallel reduction.
 * Uses TBB's parallel_reduce to efficiently process the vector in parallel,
 *  combining local results from different threads.
 * 
 * @param data Input vector containing integer values.
 * @param find_min If true, finds minimum value; if false, finds maximum value.
 * 
 * @return The minimum or maximum value found in the vector.
 */
int parallelFinMinMax(const std::vector<int> &data, bool find_min)
{
	if (find_min)
	{
		return oneapi::tbb::parallel_reduce(
			oneapi::tbb::blocked_range<size_t>(0, data.size()),
			std::numeric_limits<int>::max(),

			// Reduce operation
			[&](const oneapi::tbb::blocked_range<size_t> &r, int local_min) -> int
			{
				for (size_t i = r.begin(); i != r.end(); i++)
				{
					local_min = std::min(local_min, data[i]);
				}
				return local_min;
			},

			// Join operation
			[](int a, int b) -> int
			{
				return std::min(a, b);
			}
		);
	}
	else
	{
		return oneapi::tbb::parallel_reduce(
			oneapi::tbb::blocked_range<size_t>(0, data.size()),
			std::numeric_limits<int>::min(),
			
			// Reduce operation
			[&](const oneapi::tbb::blocked_range<size_t> &r, int local_max) -> int
			{
				for (size_t i = r.begin(); i != r.end(); i++)
				{
					local_max = std::max(local_max, data[i]);
				}
				return local_max;
			},

			// Join operation
			[](int a, int b) -> int
			{
				return std::max(a, b);
			}
		);
	}
}


/**
 * Prints a compact representation of the histogram result vector.
 * Displays the first 10 elements (or all elements if fewer than 10),
 *  with an ellipsis indicator if more elements exist.
 * 
 * @param bins Vector of histogram values to print.
 */
void printResult(const std::vector<int> &bins)
{
	std::cout << "[";
	for (size_t i = 0; i < std::min((size_t)10, bins.size()); i++)
	{
		std::cout << bins[i]
				  << (i < std::min((size_t)10, bins.size()) - 1 ? ", " : "");
	}
	if (bins.size() > 10)
		std::cout << ", ...";
	std::cout << "]" << std::endl;
}


int main()
{
	std::cout << "Initializing data..." << std::endl;
	std::vector<int> data = random_vector(DATA_SIZE);

	// Find actual min and max values in the generated data using parallel programming
	auto start_minmax = std::chrono::high_resolution_clock::now();
	int min_value = parallelFinMinMax(data, true);   // true = find minimum
	int max_value = parallelFinMinMax(data, false);  // false = find maximum
	auto end_minmax = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_minmax = end_minmax - start_minmax;
	
	std::cout << "Parallel Min/Max find time: " << diff_minmax.count() << " s" 
			  << std::endl;

	std::cout << "Data size: " << DATA_SIZE << ", Value Range: [" << min_value 
			  << ", " << max_value << "], Bins: " << NUM_BINS << std::endl;
	
	// Print the data vector
	/*std::cout << "Data vector: [";
	for (size_t i = 0; i < data.size(); i++)
	{
		std::cout << data[i] << (i < data.size() - 1 ? ", " : "");
	}
	std::cout << "]" << std::endl;*/

	std::cout << "Default concurrency: " << oneapi::tbb::info::default_concurrency() 
			  << std::endl;

	
	// Sequential Histogram
	auto start_seq = std::chrono::high_resolution_clock::now();
	std::vector<int> seq_result = sequentialHistogram(data, NUM_BINS, min_value, max_value);
	auto end_seq = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_seq = end_seq - start_seq;
	std::cout << "\nSequential Time: " << diff_seq.count() << " s" << std::endl;
	printResult(seq_result);

	
	// Parallel Histogram
	auto start_par = std::chrono::high_resolution_clock::now();
	std::vector<int> par_result = parallelHistogram(data, NUM_BINS, min_value, max_value);
	auto end_par = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff_par = end_par - start_par;
	std::cout << "\nParallel Time: " << diff_par.count() << " s" << std::endl;
	printResult(par_result);


	// Comparison
	std::cout << "\nSpeedup: " << diff_seq.count() / diff_par.count() << "x" << std::endl;
	return 0;
}
