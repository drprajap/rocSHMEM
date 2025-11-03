"""
Unit tests for rocshmem4py basic operations

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import sys
import os


# Check if running with MPI
def is_mpi_available():
    try:
        import mpi4py
        return True
    except ImportError:
        return False


def skip_if_no_mpi():
    if not is_mpi_available():
        pytest.skip("MPI not available")


class TestBasicOperations:
    """Test basic ROCSHMEM operations"""
    
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
        
        # Finalize
        rocshmem4py.rocshmem_finalize()
    
    def test_import(self):
        """Test that rocshmem4py can be imported"""
        import rocshmem4py
        assert rocshmem4py.__version__ is not None
    
    def test_my_pe(self):
        """Test rocshmem_my_pe function"""
        import rocshmem4py
        my_pe = rocshmem4py.rocshmem_my_pe()
        assert isinstance(my_pe, int)
        assert my_pe >= 0
    
    def test_n_pes(self):
        """Test rocshmem_n_pes function"""
        import rocshmem4py
        n_pes = rocshmem4py.rocshmem_n_pes()
        assert isinstance(n_pes, int)
        assert n_pes >= 1
    
    def test_pe_range(self):
        """Test that PE number is within valid range"""
        import rocshmem4py
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        assert 0 <= my_pe < n_pes
    
    def test_constants(self):
        """Test that constants are defined"""
        import rocshmem4py
        # Note: TEAM constants not yet implemented
        # assert hasattr(rocshmem4py, 'ROCSHMEM_TEAM_WORLD')
        # assert hasattr(rocshmem4py, 'ROCSHMEM_TEAM_INVALID')
        assert hasattr(rocshmem4py, 'ROCSHMEM_SUCCESS')
        assert rocshmem4py.ROCSHMEM_SUCCESS == 0


class TestMemoryOperations:
    """Test symmetric memory allocation and management"""
    
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
    
    def test_malloc_free(self):
        """Test symmetric memory allocation and deallocation"""
        import rocshmem4py
        
        size = 1024
        ptr = rocshmem4py.rocshmem_malloc(size)
        assert ptr > 0
        
        # Free the memory
        rocshmem4py.rocshmem_free(ptr)
    
    def test_symmetric_buffer_class(self):
        """Test SymmetricBuffer class"""
        import rocshmem4py
        
        size = 2048
        buf = rocshmem4py.SymmetricBuffer(size)
        
        assert buf.size == size
        assert buf.ptr > 0
        assert int(buf) == buf.ptr
        
        # Explicit free
        buf.free()
        assert buf._freed is True
    
    def test_multiple_allocations(self):
        """Test multiple symmetric memory allocations"""
        import rocshmem4py
        
        sizes = [512, 1024, 2048, 4096]
        buffers = []
        
        for size in sizes:
            buf = rocshmem4py.SymmetricBuffer(size)
            assert buf.size == size
            buffers.append(buf)
        
        # Free all buffers
        for buf in buffers:
            buf.free()
    
    def test_rocshmem_ptr(self):
        """Test rocshmem_ptr function"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        
        # Allocate symmetric memory
        buf = rocshmem4py.SymmetricBuffer(1024)
        
        # Get pointer to local PE (should return valid pointer or same pointer)
        local_ptr = rocshmem4py.rocshmem_ptr(buf.ptr, my_pe)
        assert local_ptr >= 0
        
        # Test SymmetricBuffer.get_remote_ptr method
        remote_ptr = buf.get_remote_ptr(my_pe)
        assert remote_ptr >= 0
        
        # Clean up
        buf.free()


class TestSynchronization:
    """Test synchronization operations"""
    
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
    
    def test_barrier_all(self):
        """Test barrier synchronization"""
        import rocshmem4py
        
        # This should complete without error
        rocshmem4py.rocshmem_barrier_all()
    
    def test_fence(self):
        """Test fence operation"""
        import rocshmem4py
        
        # This should complete without error
        rocshmem4py.rocshmem_fence()
    
    def test_quiet(self):
        """Test quiet operation"""
        import rocshmem4py
        
        # This should complete without error
        rocshmem4py.rocshmem_quiet()


@pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="Requires at least 2 PEs"
)
class TestDataTransfer:
    """Test data transfer operations (requires multiple PEs)"""
    
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
    
    def test_rocshmem_ptr(self):
        """Test getting remote symmetric pointer"""
        import rocshmem4py
        
        my_pe = rocshmem4py.rocshmem_my_pe()
        n_pes = rocshmem4py.rocshmem_n_pes()
        
        if n_pes < 2:
            pytest.skip("Requires at least 2 PEs")
        
        # Allocate symmetric memory
        buf = rocshmem4py.SymmetricBuffer(1024)
        
        # Get remote pointer to PE 0
        remote_pe = 0 if my_pe != 0 else 1
        remote_ptr = rocshmem4py.rocshmem_ptr(buf.ptr, remote_pe)
        
        # Remote pointer should be valid
        assert remote_ptr > 0
        
        buf.free()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
