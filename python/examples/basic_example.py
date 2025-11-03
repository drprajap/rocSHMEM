#!/usr/bin/env python3
"""
Basic example demonstrating rocshmem4py usage

This example shows:
- Initialization with MPI
- Getting PE information
- Allocating symmetric memory
- Barrier synchronization
- Finalization

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""

import sys
from mpi4py import MPI
import rocshmem4py


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    
    print(f"MPI Rank {mpi_rank}/{mpi_size}: Starting ROCSHMEM initialization")
    
    # Initialize ROCSHMEM with MPI
    rocshmem4py.init_with_mpi(comm)
    
    # Get PE information
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    
    print(f"PE {my_pe}/{n_pes}: ROCSHMEM initialized successfully")
    
    # Allocate symmetric memory
    buffer_size = 1024
    symm_buffer = rocshmem4py.SymmetricBuffer(buffer_size)
    
    print(f"PE {my_pe}: Allocated {buffer_size} bytes of symmetric memory at 0x{symm_buffer.ptr:x}")
    
    # Synchronize all PEs
    rocshmem4py.rocshmem_barrier_all()
    
    if my_pe == 0:
        print("All PEs have reached the barrier")
    
    # Free symmetric memory
    symm_buffer.free()
    
    # Finalize ROCSHMEM
    rocshmem4py.rocshmem_finalize()
    
    if my_pe == 0:
        print("ROCSHMEM finalized successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
