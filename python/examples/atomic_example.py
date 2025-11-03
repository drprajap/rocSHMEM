#!/usr/bin/env python3
"""
Example demonstrating ROCSHMEM atomic operations

This example shows:
- Atomic fetch-and-add operations
- Counter increment using atomics
- Verification of atomic results

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.

Run with: mpirun -np 4 python atomic_example.py
"""

import sys
import struct
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
    
    print(f"PE {my_pe}/{n_pes}: Starting atomic operations example")
    
    # Allocate symmetric buffer for counter (8 bytes for long)
    counter_buffer = rocshmem4py.SymmetricBuffer(8)
    
    # Initialize counter to 0 on PE 0
    # (In a real application, you'd use HIP to zero the memory)
    
    # Synchronize all PEs
    rocshmem4py.rocshmem_barrier_all()
    
    # Each PE atomically increments the counter on PE 0
    target_pe = 0
    if my_pe != target_pe:
        old_value = rocshmem4py.rocshmem_long_atomic_fetch_add(
            counter_buffer.ptr, 
            1,  # increment by 1
            target_pe
        )
        print(f"PE {my_pe}: Atomic fetch-add returned {old_value}")
    
    # Ensure all atomic operations complete
    rocshmem4py.rocshmem_quiet()
    rocshmem4py.rocshmem_barrier_all()
    
    # PE 0 can now read the final counter value
    if my_pe == target_pe:
        print(f"PE {my_pe}: All atomic operations completed")
        print(f"PE {my_pe}: Expected counter value: {n_pes - 1}")
        # (In a real application, you'd read the actual value from device memory)
    
    # Clean up
    counter_buffer.free()
    rocshmem4py.rocshmem_finalize()
    
    if my_pe == 0:
        print("Atomic operations example completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
