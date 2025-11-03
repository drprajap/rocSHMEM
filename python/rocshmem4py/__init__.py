"""
rocshmem4py: Python bindings for ROCm Shared Memory (ROCSHMEM) library

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
from typing import Sequence, Optional, Any

__version__ = "0.1.0"
__author__ = "Advanced Micro Devices, Inc."

# Try to import the C++ extension module
try:
    from _rocshmem4py import *
except ImportError as e:
    print(
        "Failed to import _rocshmem4py extension. "
        "Please ensure ROCSHMEM is properly installed and "
        "LD_LIBRARY_PATH includes the ROCSHMEM library path.",
        file=sys.stderr
    )
    raise e

# Re-export core functions and constants
from _rocshmem4py import (
    rocshmem_init,
    rocshmem_finalize,
    rocshmem_my_pe,
    rocshmem_n_pes,
    rocshmem_malloc,
    rocshmem_free,
    rocshmem_ptr,
    rocshmem_barrier_all,
    rocshmem_fence,
    rocshmem_quiet,
    rocshmem_get_uniqueid,
    rocshmem_init_attr,
    rocshmem_putmem,
    rocshmem_getmem,
    rocshmem_putmem_nbi,
    rocshmem_getmem_nbi,
    rocshmem_int_atomic_fetch_add,
    rocshmem_long_atomic_fetch_add,
    rocshmem_int_atomic_compare_swap,
    ROCSHMEM_SUCCESS,
)

__all__ = [
    '__version__',
    # Core functions
    'rocshmem_init',
    'rocshmem_finalize',
    'rocshmem_my_pe',
    'rocshmem_n_pes',
    'rocshmem_malloc',
    'rocshmem_free',
    'rocshmem_ptr',
    'rocshmem_barrier_all',
    'rocshmem_fence',
    'rocshmem_quiet',
    'rocshmem_get_uniqueid',
    'rocshmem_init_attr',
    # Data transfer
    'rocshmem_putmem',
    'rocshmem_getmem',
    'rocshmem_putmem_nbi',
    'rocshmem_getmem_nbi',
    # Atomic operations
    'rocshmem_int_atomic_fetch_add',
    'rocshmem_long_atomic_fetch_add',
    'rocshmem_int_atomic_compare_swap',
    # Constants
    'ROCSHMEM_SUCCESS',
    # High-level API
    'SymmetricBuffer',
    'init_with_mpi',
    'init_with_torch',
]


class SymmetricBuffer:
    """
    A Python wrapper for symmetric memory allocated with rocshmem_malloc.
    
    This class manages the lifecycle of symmetric memory and provides
    a convenient interface for accessing remote memory.
    """
    
    def __init__(self, size: int):
        """
        Allocate symmetric memory.
        
        Args:
            size: Size in bytes to allocate
        """
        self.size = size
        self.ptr = rocshmem_malloc(size)
        self._freed = False
        
    def __del__(self):
        """Free symmetric memory when the object is garbage collected."""
        if not self._freed:
            self.free()
    
    def free(self):
        """Explicitly free the symmetric memory."""
        if not self._freed:
            rocshmem_free(self.ptr)
            self._freed = True
    
    def get_remote_ptr(self, pe: int) -> int:
        """
        Get a pointer to this symmetric memory on a remote PE.
        
        Args:
            pe: The PE number to get the pointer from
            
        Returns:
            Pointer to the symmetric memory on the remote PE
        """
        return rocshmem_ptr(self.ptr, pe)
    
    def put(self, source_ptr: int, nelems: int, pe: int):
        """
        Put data to a remote PE.
        
        Args:
            source_ptr: Source pointer
            nelems: Number of bytes to transfer
            pe: Destination PE
        """
        rocshmem_putmem(self.ptr, source_ptr, nelems, pe)
    
    def get(self, source_ptr: int, nelems: int, pe: int):
        """
        Get data from a remote PE.
        
        Args:
            source_ptr: Source pointer on remote PE
            nelems: Number of bytes to transfer
            pe: Source PE
        """
        rocshmem_getmem(self.ptr, source_ptr, nelems, pe)
    
    def __int__(self) -> int:
        """Return the pointer as an integer."""
        return self.ptr
    
    def __repr__(self) -> str:
        status = "freed" if self._freed else f"size={self.size}"
        return f"SymmetricBuffer(ptr=0x{self.ptr:x}, {status})"


def init_with_mpi(mpi_comm: Optional[Any] = None):
    """
    Initialize ROCSHMEM using MPI for coordination.
    
    This function uses MPI to coordinate the initialization of ROCSHMEM
    across multiple processes. It is particularly useful when integrating
    ROCSHMEM with existing MPI applications.
    
    Args:
        mpi_comm: MPI communicator (optional, defaults to MPI.COMM_WORLD)
    
    Example:
        >>> from mpi4py import MPI
        >>> import rocshmem4py
        >>> rocshmem4py.init_with_mpi(MPI.COMM_WORLD)
        >>> my_pe = rocshmem4py.rocshmem_my_pe()
        >>> n_pes = rocshmem4py.rocshmem_n_pes()
        >>> print(f"PE {my_pe} of {n_pes}")
    """
    try:
        from mpi4py import MPI
        if mpi_comm is None:
            mpi_comm = MPI.COMM_WORLD
        
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
        
        # Get unique ID from rank 0
        if rank == 0:
            unique_id = rocshmem_get_uniqueid()
        else:
            unique_id = None
        
        # Broadcast unique ID to all ranks
        unique_id = mpi_comm.bcast(unique_id, root=0)
        
        # Initialize ROCSHMEM with the unique ID
        rocshmem_init_attr(rank, size, unique_id)
        
        # Barrier to ensure all ranks are initialized
        mpi_comm.Barrier()
        
    except ImportError:
        raise ImportError(
            "mpi4py is required for init_with_mpi(). "
            "Install it with: pip install mpi4py"
        )


def init_with_torch(backend: str = 'nccl', init_method: Optional[str] = None):
    """
    Initialize ROCSHMEM using PyTorch distributed for coordination.
    
    This function uses torch.distributed to coordinate the initialization of ROCSHMEM
    across multiple processes. It works seamlessly with PyTorch distributed training.
    
    Note: torch.distributed.init_process_group() must be called before this function,
    or you can let this function initialize it by providing backend and init_method.
    
    Args:
        backend: PyTorch distributed backend ('nccl', 'gloo', etc.)
                Only used if torch.distributed is not already initialized
        init_method: Initialization method (e.g., 'env://', 'tcp://...')
                    Only used if torch.distributed is not already initialized
    
    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> import rocshmem4py
        >>> 
        >>> # Option 1: Initialize torch.distributed first
        >>> dist.init_process_group(backend='nccl')
        >>> rocshmem4py.init_with_torch()
        >>> 
        >>> # Option 2: Let rocshmem4py initialize torch.distributed
        >>> rocshmem4py.init_with_torch(backend='nccl', init_method='env://')
        >>> 
        >>> my_pe = rocshmem4py.rocshmem_my_pe()
        >>> n_pes = rocshmem4py.rocshmem_n_pes()
        >>> print(f"PE {my_pe} of {n_pes}")
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        raise ImportError(
            "PyTorch is required for init_with_torch(). "
            "Install it with: pip install torch"
        )
    
    # Initialize torch.distributed if not already initialized
    if not dist.is_initialized():
        if init_method is None:
            # Try to use environment variables
            init_method = 'env://'
        
        try:
            dist.init_process_group(backend=backend, init_method=init_method)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize torch.distributed: {e}\n"
                "Make sure RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT "
                "environment variables are set, or provide init_method explicitly."
            )
    
    # Get rank and world size from torch.distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Get unique ID from rank 0
    if rank == 0:
        unique_id_bytes = rocshmem_get_uniqueid()
        # Convert bytes to tensor for broadcasting
        unique_id_tensor = torch.frombuffer(unique_id_bytes, dtype=torch.uint8).clone()
    else:
        # Create empty tensor to receive broadcast
        unique_id_tensor = torch.empty(128, dtype=torch.uint8)
    
    # Ensure tensor is on CPU for broadcast
    unique_id_tensor = unique_id_tensor.cpu()
    
    # Broadcast unique ID to all ranks
    dist.broadcast(unique_id_tensor, src=0)
    
    # Synchronize
    dist.barrier()
    
    # Convert tensor back to bytes
    unique_id = unique_id_tensor.numpy().tobytes()
    
    # Initialize ROCSHMEM with the unique ID
    rocshmem_init_attr(rank, world_size, unique_id)
    
    # Final barrier to ensure all ranks are initialized
    dist.barrier()
