# rocshmem4py: Python Bindings for ROCSHMEM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

`rocshmem4py` provides Python bindings for the ROCm Shared Memory (ROCSHMEM) library, enabling high-performance communication for GPU-accelerated applications on AMD ROCm platforms.

## Features

- **Complete ROCSHMEM API Coverage**: Access to all major ROCSHMEM operations
- **Pythonic Interface**: High-level Python wrappers with memory management
- **MPI Integration**: Seamless initialization with MPI via mpi4py
- **PyTorch Integration**: Initialize with torch.distributed for seamless PyTorch workflows
- **Zero-Copy Operations**: Direct access to GPU memory
- **Type Safety**: Strong typing with comprehensive error checking

## Installation

### Prerequisites

- AMD ROCm 5.0+ with HIP support
- ROCSHMEM library installed
- Python 3.7 or later
- CMake 3.16 or later
- pybind11 2.6.0 or later
- MPI implementation (OpenMPI, MPICH, etc.) - optional but recommended

### Environment Setup

Before building, set the following environment variables:

```bash
export ROCM_PATH=/opt/rocm              # Path to ROCm installation
export ROCSHMEM_HOME=/path/to/rocshmem  # Path to ROCSHMEM installation
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

### Build from Source

```bash
cd /path/to/rocSHMEM/python

# Install build dependencies
pip install pybind11 cmake

# Build and install
pip install -e .

# Or build with setuptools directly
python setup.py build_ext --inplace
python setup.py install
```

### Install from PyPI (Coming Soon)

```bash
pip install rocshmem4py
```

## Quick Start

### Basic Example with MPI

```python
from mpi4py import MPI
import rocshmem4py

# Initialize ROCSHMEM with MPI
rocshmem4py.init_with_mpi(MPI.COMM_WORLD)

# Get PE information
my_pe = rocshmem4py.rocshmem_my_pe()
n_pes = rocshmem4py.rocshmem_n_pes()
print(f"Hello from PE {my_pe} of {n_pes}")

# Allocate symmetric memory
buffer = rocshmem4py.SymmetricBuffer(1024)

# Synchronize all PEs
rocshmem4py.rocshmem_barrier_all()

# Clean up
buffer.free()
rocshmem4py.rocshmem_finalize()
```

Run with MPI:
```bash
mpirun -np 2 python example.py
```

### PyTorch Integration Example

```python
import torch
import torch.distributed as dist
import rocshmem4py

# Initialize PyTorch distributed
dist.init_process_group(backend='nccl')

# Initialize ROCSHMEM using PyTorch distributed
rocshmem4py.init_with_torch()

# Get PE information (matches torch rank)
my_pe = rocshmem4py.rocshmem_my_pe()
n_pes = rocshmem4py.rocshmem_n_pes()
assert my_pe == dist.get_rank()
assert n_pes == dist.get_world_size()

# Allocate symmetric memory
buffer = rocshmem4py.SymmetricBuffer(1024)

# Use with PyTorch tensors
device = torch.device(f'cuda:{torch.cuda.current_device()}')
tensor = torch.ones(256, device=device) * my_pe

# Synchronize
rocshmem4py.rocshmem_barrier_all()

# Clean up
buffer.free()
rocshmem4py.rocshmem_finalize()
dist.destroy_process_group()
```

Run with PyTorch:
```bash
torchrun --nproc_per_node=2 pytorch_example.py
```

### Data Transfer Example

```python
import rocshmem4py

# Initialize
rocshmem4py.rocshmem_init()

my_pe = rocshmem4py.rocshmem_my_pe()
n_pes = rocshmem4py.rocshmem_n_pes()

# Allocate symmetric buffer
buffer = rocshmem4py.SymmetricBuffer(256)

# Put data to next PE
if my_pe < n_pes - 1:
    target_pe = my_pe + 1
    rocshmem4py.rocshmem_putmem(buffer.ptr, buffer.ptr, 256, target_pe)
    rocshmem4py.rocshmem_fence()

# Synchronize
rocshmem4py.rocshmem_barrier_all()

# Clean up
buffer.free()
rocshmem4py.rocshmem_finalize()
```

## API Reference

### Initialization and Finalization

- `rocshmem_init()` - Initialize ROCSHMEM
- `rocshmem_finalize()` - Finalize ROCSHMEM
- `rocshmem_init_attr(rank, nranks, uniqueid)` - Initialize with attributes
- `init_with_mpi(comm)` - Initialize using MPI communicator (high-level)

### PE Query Functions

- `rocshmem_my_pe()` → `int` - Get the PE number of the calling PE
- `rocshmem_n_pes()` → `int` - Get the total number of PEs
- `rocshmem_team_my_pe(team)` → `int` - Get PE number within a team
- `rocshmem_team_n_pes(team)` → `int` - Get number of PEs in a team

### Memory Management

- `rocshmem_malloc(size)` → `int` - Allocate symmetric memory
- `rocshmem_free(ptr)` - Free symmetric memory
- `rocshmem_ptr(dest, pe)` → `int` - Get remote symmetric pointer
- `SymmetricBuffer(size)` - High-level symmetric memory wrapper class

### Data Transfer Operations

#### Blocking Operations

- `rocshmem_putmem(dest, source, nelems, pe)` - Put data to remote PE
- `rocshmem_getmem(dest, source, nelems, pe)` - Get data from remote PE

#### Non-blocking Operations

- `rocshmem_putmem_nbi(dest, source, nelems, pe)` - Non-blocking put
- `rocshmem_getmem_nbi(dest, source, nelems, pe)` - Non-blocking get

### Synchronization

- `rocshmem_barrier_all()` - Barrier across all PEs
- `rocshmem_fence()` - Ensure ordering of memory operations
- `rocshmem_quiet()` - Wait for completion of all outstanding operations

### Atomic Operations

- `rocshmem_int_atomic_fetch_add(dest, value, pe)` → `int`
- `rocshmem_long_atomic_fetch_add(dest, value, pe)` → `int`
- `rocshmem_int_atomic_compare_swap(dest, cond, value, pe)` → `int`

### Collective Operations

#### Reductions

- `rocshmem_int_sum_reduce(team, dest, source, nreduce)`
- `rocshmem_long_sum_reduce(team, dest, source, nreduce)`
- `rocshmem_float_sum_reduce(team, dest, source, nreduce)`
- `rocshmem_double_sum_reduce(team, dest, source, nreduce)`

#### Other Collectives

- `rocshmem_broadcastmem(team, dest, source, nelems, pe_root)`
- `rocshmem_alltoallmem(team, dest, source, nelems)`

### Constants

- `ROCSHMEM_TEAM_WORLD` - Team containing all PEs
- `ROCSHMEM_TEAM_INVALID` - Invalid team identifier
- `ROCSHMEM_SUCCESS` - Success status code

## Examples

See the `examples/` directory for complete examples:

- `basic_example.py` - Basic initialization and memory allocation
- `put_get_example.py` - Data transfer operations
- `atomic_example.py` - Atomic operations
- `reduction_example.py` - Collective reduction operations

### Running Examples

```bash
# Single process
python examples/basic_example.py

# Multiple processes with MPI
mpirun -np 4 python examples/put_get_example.py
mpirun -np 4 python examples/atomic_example.py
mpirun -np 4 python examples/reduction_example.py
```

## Testing

The package includes comprehensive unit tests using pytest.

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_basic.py -v

# Run with MPI (for multi-PE tests)
mpirun -np 2 pytest tests/ -v
```

### Test Categories

- `test_basic.py` - Basic operations (initialization, PE queries, memory)
- `test_collective.py` - Collective and atomic operations (requires multiple PEs)

## Advanced Usage

### Using with PyTorch

```python
import torch
import rocshmem4py
from mpi4py import MPI

# Initialize
rocshmem4py.init_with_mpi(MPI.COMM_WORLD)

# Allocate symmetric buffer
buffer = rocshmem4py.SymmetricBuffer(1024)

# Create a PyTorch tensor view (requires CUDA array interface implementation)
# This is a placeholder - actual implementation would depend on your use case

# Clean up
buffer.free()
rocshmem4py.rocshmem_finalize()
```

### Custom Memory Management

```python
import rocshmem4py

# Manual memory allocation
ptr = rocshmem4py.rocshmem_malloc(2048)

# Use the pointer for ROCSHMEM operations
# ...

# Manual deallocation
rocshmem4py.rocshmem_free(ptr)
```

## Troubleshooting

### Import Error

```
ImportError: Failed to import _rocshmem4py extension
```

**Solution**: Ensure ROCSHMEM library is in `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$LD_LIBRARY_PATH
```

### CMake Not Finding ROCSHMEM

**Solution**: Set `ROCSHMEM_HOME` environment variable:
```bash
export ROCSHMEM_HOME=/path/to/rocshmem/install
```

### MPI Initialization Issues

**Solution**: Ensure mpi4py is properly installed and MPI is working:
```bash
pip install mpi4py
mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

## Acknowledgments

This project builds upon the ROCSHMEM library developed by AMD and the Python bindings initially developed for the Triton-distributed project.
