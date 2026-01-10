# Accumulative Histogram using Intel TBB

A parallel implementation of an accumulative histogram algorithm using the Intel Threading Building Blocks (TBB) library. This project compares sequential and parallel approaches for computing cumulative histograms.

## Description

This program computes an accumulative (cumulative) histogram from a vector of numerical values. Given a dataset and a predefined number of bins:

1. **Frequency Counting**: Values are classified into bins based on their range
2. **Accumulation**: A prefix sum is computed to convert frequencies into cumulative counts

The implementation provides both:
- **Sequential version**: Traditional single-threaded approach
- **Parallel version**: Multi-threaded implementation using Intel TBB's `parallel_reduce` and `parallel_scan`

### Algorithm Details

- **Data Generation**: Random integers in range [-500, 1000] (supports negative values)
- **Min/Max Detection**: Uses `parallel_reduce` to find the actual data range
- **Parallel Binning**: Uses `parallel_reduce` to count frequencies across multiple threads
- **Parallel Accumulation**: Uses `parallel_scan` for the prefix sum operation

## Configuration

Default parameters (can be modified in `main.cpp`):

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `NUM_BINS` | 10 | Number of histogram bins |
| `DATA_SIZE` | 10,000,000 | Size of the input data vector |
| `MIN_VALUE` | -500 | Minimum value for random generation |
| `MAX_VALUE` | 1000 | Maximum value for random generation |

## Requirements

- C++17 or later
- Intel oneAPI TBB library

## How to Run

### Option 1: Using Docker (Recommended)

Run inside the oneTBB Docker container:

```bash
docker run --rm -it -v $(pwd):/project mfisherman/onetbb
```

Inside the container:

```bash
cd /project
g++ -g -std=c++17 main.cpp -pthread -ltbb -o histogram
./histogram
```

### Option 2: Local Installation with oneAPI

If you have Intel oneAPI installed locally:

```bash
# Source the oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Compile and run
g++ -g -std=c++17 main.cpp -pthread -ltbb -o histogram
./histogram
```

## Expected Output

```
Initializing data...
Parallel Min/Max find time: 0.011965 s
Data size: 10000000, Value Range: [-500, 1000], Bins: 10
Default concurrency: 22

Histogram Data by Bins:
  Bin 0 [-500, -350): 1006147
  Bin 1 [-350, -200): 998833
  Bin 2 [-200, -50): 999028
  Bin 3 [-50, 100): 999942
  Bin 4 [100, 250): 998327
  Bin 5 [250, 400): 998219
  Bin 6 [400, 550): 1000250
  Bin 7 [550, 700): 998862
  Bin 8 [700, 850): 1000026
  Bin 9 [850, 1001): 1000366

Sequential Time: 0.0625173 s
[1006147, 2004980, 3004008, 4003950, 5002277, 6000496, 7000746, 7999608, 8999634, 10000000]

Parallel Time: 0.00678759 s
[1006147, 2004980, 3004008, 4003950, 5002277, 6000496, 7000746, 7999608, 8999634, 10000000]

Speedup: 9.21052x

```

## Implementation Highlights

### Sequential Histogram
- Simple iterative approach for binning
- Linear prefix sum for accumulation
- Time complexity: O(n + k) where n = data size, k = number of bins

### Parallel Histogram
- **`parallel_reduce`**: Divides data into chunks, each thread maintains local bin counts, then combines results
- **`parallel_scan`**: Performs parallel prefix sum for accumulation
- Achieves speedup proportional to available CPU cores

## Project Structure

```
.
├── main.cpp    # Main source code with sequential and parallel implementations
└── README.md   # This file
```
