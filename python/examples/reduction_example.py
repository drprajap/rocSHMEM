#!/usr/bin/env python3
"""
Example demonstrating ROCSHMEM reduction operations

This example shows:
- Sum reduction across all PEs
- Using collective operations with ROCSHMEM_TEAM_WORLD
- Synchronization before and after collectives

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.

Run with: mpirun -np 4 python reduction_example.py
"""

import sys
from mpi4py import MPI
import rocshmem4py


def main():
    # Initialize
    comm = MPI.COMM_WORLD
    rocshmem4py.init_with_mpi(comm)
    
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    
    if n_pes < 2:
        if my_pe == 0:
            print("This example requires at least 2 PEs")
        rocshmem4py.rocshmem_finalize()
        return
    
    print(f"PE {my_pe}/{n_pes}: Starting reduction example")
    
    # Number of elements to reduce
    nelems = 10
    
    # Allocate symmetric buffers for source and destination
    # Each integer is 4 bytes
    src_buffer = rocshmem4py.SymmetricBuffer(nelems * 4)
    dst_buffer = rocshmem4py.SymmetricBuffer(nelems * 4)
    
    # In a real application, you would initialize src_buffer with values
    # For this example, we assume each PE has initialized its buffer with its PE number
    print(f"PE {my_pe}: Buffers allocated, performing integer sum reduction")
    
    # Synchronize before reduction
    rocshmem4py.rocshmem_barrier_all()
    
    # Perform integer sum reduction
    rocshmem4py.rocshmem_int_sum_reduce(
        rocshmem4py.ROCSHMEM_TEAM_WORLD,
        dst_buffer.ptr,
        src_buffer.ptr,
        nelems
    )
    
    # Synchronize after reduction
    rocshmem4py.rocshmem_barrier_all()
    
    print(f"PE {my_pe}: Integer sum reduction completed")
    
    # Now demonstrate float reduction
    float_src = rocshmem4py.SymmetricBuffer(nelems * 4)  # 4 bytes per float
    float_dst = rocshmem4py.SymmetricBuffer(nelems * 4)
    
    rocshmem4py.rocshmem_barrier_all()
    
    rocshmem4py.rocshmem_float_sum_reduce(
        rocshmem4py.ROCSHMEM_TEAM_WORLD,
        float_dst.ptr,
        float_src.ptr,
        nelems
    )
    
    rocshmem4py.rocshmem_barrier_all()
    print(f"PE {my_pe}: Float sum reduction completed")
    
    # Demonstrate double reduction
    double_src = rocshmem4py.SymmetricBuffer(nelems * 8)  # 8 bytes per double
    double_dst = rocshmem4py.SymmetricBuffer(nelems * 8)
    
    rocshmem4py.rocshmem_barrier_all()
    
    rocshmem4py.rocshmem_double_sum_reduce(
        rocshmem4py.ROCSHMEM_TEAM_WORLD,
        double_dst.ptr,
        double_src.ptr,
        nelems
    )
    
    rocshmem4py.rocshmem_barrier_all()
    print(f"PE {my_pe}: Double sum reduction completed")
    
    # Clean up
    src_buffer.free()
    dst_buffer.free()
    float_src.free()
    float_dst.free()
    double_src.free()
    double_dst.free()
    
    rocshmem4py.rocshmem_finalize()
    
    if my_pe == 0:
        print("Reduction example completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
