# Quick Start Guide for rocshmem4py

This guide will help you quickly build and test rocshmem4py.

## Prerequisites

1. **ROCm**: Install ROCm 5.0 or later
2. **ROCSHMEM**: Build and install ROCSHMEM library
3. **Python**: Python 3.7 or later
4. **MPI** (optional but recommended): OpenMPI or MPICH

## Step 1: Set Environment Variables

```bash
# Set ROCSHMEM installation path
export ROCSHMEM_HOME=/path/to/rocshmem/install

# Set ROCm path (optional, defaults to /opt/rocm)
export ROCM_PATH=/opt/rocm

# Update library path
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

## Step 2: Build rocshmem4py

```bash
cd /path/to/rocSHMEM/python

# Simple build
./build.sh

# Or build in debug mode
./build.sh --debug

# Or clean build
./build.sh --clean
```

## Step 3: Run Tests

```bash
# Run all tests
./run_tests.sh

# Run with more MPI processes
./run_tests.sh -np 4

# Run specific test file
./run_tests.sh -f test_basic.py -v

# Or use pytest directly
pytest tests/ -v
```

## Step 4: Try Examples

```bash
# Single process example
python3 examples/basic_example.py

# Multi-process examples (requires MPI)
mpirun -np 2 python3 examples/put_get_example.py
mpirun -np 4 python3 examples/atomic_example.py
mpirun -np 4 python3 examples/reduction_example.py
```

## Hello World Example

Create a file `hello_rocshmem.py`:

```python
from mpi4py import MPI
import rocshmem4py

# Initialize
rocshmem4py.init_with_mpi(MPI.COMM_WORLD)

# Get PE info
my_pe = rocshmem4py.rocshmem_my_pe()
n_pes = rocshmem4py.rocshmem_n_pes()

print(f"Hello from PE {my_pe} of {n_pes}")

# Synchronize
rocshmem4py.rocshmem_barrier_all()

# Finalize
rocshmem4py.rocshmem_finalize()
```

Run it:
```bash
mpirun -np 4 python3 hello_rocshmem.py
```

## Building PyPI Wheel

To build a wheel for distribution:

```bash
# Install build tools
pip install build wheel

# Build wheel
python3 -m build

# The wheel will be in dist/
ls dist/
```
