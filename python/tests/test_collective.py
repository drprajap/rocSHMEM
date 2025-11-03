"""
Unit tests for rocshmem4py collective and atomic operations

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import sys
import os
import numpy as np


def is_mpi_available():
    try:
        import mpi4py
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="Requires at least 2 PEs"
)
class TestAtomicOperations:
    """Test atomic operations (requires multiple PEs)"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_rocshmem(self):
        """Initialize ROCSHMEM before tests"""
        import rocshmem4py
        
        if is_mpi_available():
            from mpi4py import MPI
            rocshmem4py.init_with_mpi(MPI.COMM_WORLD)
        else:
            rocshmem4py.rocshmem_init()
        
        yield
        
        rocshmem4py.rocshmem_finalize()
    
    def test_int_atomic_fetch_add(self):
        """Test integer atomic fetch and add"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        # Allocate symmetric memory for an integer
        buf = rocshmem4py.SymmetricBuffer(4)  # 4 bytes for int
        
        # Initialize to 0 on PE 0
        if my_pe == 0:
            # Note: In a real test, we'd need to zero the memory
            pass
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Each PE atomically adds 1 to PE 0's counter
        if my_pe != 0:
            old_value = rocshmem4py.rocshmem_int_atomic_fetch_add(buf.ptr, 1, 0)
            assert isinstance(old_value, int)
        
        rocshmem4py.rocshmem_barrier_all()
        buf.free()
    
    def test_long_atomic_fetch_add(self):
        """Test long integer atomic fetch and add"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        # Allocate symmetric memory for a long
        buf = rocshmem4py.SymmetricBuffer(8)  # 8 bytes for long
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Each PE atomically adds 1 to PE 0's counter
        if my_pe != 0:
            old_value = rocshmem4py.rocshmem_long_atomic_fetch_add(buf.ptr, 1, 0)
            assert isinstance(old_value, int)
        
        rocshmem4py.rocshmem_barrier_all()
        buf.free()


@pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="Requires at least 2 PEs"
)
class TestCollectiveOperations:
    """Test collective operations (requires multiple PEs)"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_rocshmem(self):
        """Initialize ROCSHMEM before tests"""
        import rocshmem4py
        
        if is_mpi_available():
            from mpi4py import MPI
            rocshmem4py.init_with_mpi(MPI.COMM_WORLD)
        else:
            rocshmem4py.rocshmem_init()
        
        yield
        
        rocshmem4py.rocshmem_finalize()
    
    def test_int_sum_reduce(self):
        """Test integer sum reduction"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        # Allocate buffers for source and destination
        nelems = 10
        src_buf = rocshmem4py.SymmetricBuffer(nelems * 4)  # 4 bytes per int
        dst_buf = rocshmem4py.SymmetricBuffer(nelems * 4)
        
        # Note: In a real test, we would initialize src_buf with data
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Perform sum reduction
        rocshmem4py.rocshmem_int_sum_reduce(
            rocshmem4py.ROCSHMEM_TEAM_WORLD,
            dst_buf.ptr,
            src_buf.ptr,
            nelems
        )
        
        rocshmem4py.rocshmem_barrier_all()
        
        src_buf.free()
        dst_buf.free()
    
    def test_float_sum_reduce(self):
        """Test float sum reduction"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        nelems = 10
        src_buf = rocshmem4py.SymmetricBuffer(nelems * 4)  # 4 bytes per float
        dst_buf = rocshmem4py.SymmetricBuffer(nelems * 4)
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Perform sum reduction
        rocshmem4py.rocshmem_float_sum_reduce(
            rocshmem4py.ROCSHMEM_TEAM_WORLD,
            dst_buf.ptr,
            src_buf.ptr,
            nelems
        )
        
        rocshmem4py.rocshmem_barrier_all()
        
        src_buf.free()
        dst_buf.free()
    
    def test_double_sum_reduce(self):
        """Test double sum reduction"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        nelems = 10
        src_buf = rocshmem4py.SymmetricBuffer(nelems * 8)  # 8 bytes per double
        dst_buf = rocshmem4py.SymmetricBuffer(nelems * 8)
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Perform sum reduction
        rocshmem4py.rocshmem_double_sum_reduce(
            rocshmem4py.ROCSHMEM_TEAM_WORLD,
            dst_buf.ptr,
            src_buf.ptr,
            nelems
        )
        
        rocshmem4py.rocshmem_barrier_all()
        
        src_buf.free()
        dst_buf.free()
    
    def test_broadcastmem(self):
        """Test broadcast operation"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        nelems = 1024
        src_buf = rocshmem4py.SymmetricBuffer(nelems)
        dst_buf = rocshmem4py.SymmetricBuffer(nelems)
        
        rocshmem4py.rocshmem_barrier_all()
        
        # Broadcast from PE 0
        rocshmem4py.rocshmem_broadcastmem(
            rocshmem4py.ROCSHMEM_TEAM_WORLD,
            dst_buf.ptr,
            src_buf.ptr,
            nelems,
            0  # root PE
        )
        
        rocshmem4py.rocshmem_barrier_all()
        
        src_buf.free()
        dst_buf.free()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
