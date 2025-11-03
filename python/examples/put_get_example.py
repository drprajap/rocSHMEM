#!/usr/bin/env python3
"""
Example demonstrating ROCSHMEM put/get operations

This example shows:
- Data transfer between PEs using put/get
- Remote memory access
- Synchronization with fence and barrier

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.

Run with: mpirun -np 2 python put_get_example.py
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
    
    print(f"PE {my_pe}: Starting put/get example")
    
    # Allocate symmetric buffers
    buffer_size = 64
    local_buffer = rocshmem4py.SymmetricBuffer(buffer_size)
    
    # PE 0 will put data to PE 1
    if my_pe == 0:
        print(f"PE {my_pe}: Preparing to send data to PE 1")
        # In a real application, you would initialize the buffer with data here
        # For now, we just demonstrate the API
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Put data to PE 1
        target_pe = 1
        print(f"PE {my_pe}: Sending {buffer_size} bytes to PE {target_pe}")
        rocshmem4py.rocshmem_putmem(local_buffer.ptr, local_buffer.ptr, buffer_size, target_pe)
        
        # Ensure the put completes
        rocshmem4py.rocshmem_fence()
        print(f"PE {my_pe}: Put operation completed")
        
    elif my_pe == 1:
        print(f"PE {my_pe}: Waiting to receive data from PE 0")
        rocshmem4py.rocshmem_barrier_all()
        
        # Wait for data using quiet (in real code, you might use fence or barrier)
        rocshmem4py.rocshmem_quiet()
        print(f"PE {my_pe}: Data received")
    
    # Synchronize all PEs
    rocshmem4py.rocshmem_barrier_all()
    
    # Now PE 1 will get data from PE 0
    if my_pe == 1:
        print(f"PE {my_pe}: Getting data from PE 0")
        source_pe = 0
        rocshmem4py.rocshmem_getmem(local_buffer.ptr, local_buffer.ptr, buffer_size, source_pe)
        rocshmem4py.rocshmem_quiet()
        print(f"PE {my_pe}: Get operation completed")
    
    # Synchronize
    rocshmem4py.rocshmem_barrier_all()
    
    # Clean up
    local_buffer.free()
    rocshmem4py.rocshmem_finalize()
    
    if my_pe == 0:
        print("Example completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
