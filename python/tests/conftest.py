"""
pytest configuration for rocshmem4py tests

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""

import pytest
import sys
import os


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "mpi: mark test as requiring MPI"
    )
    config.addinivalue_line(
        "markers", "multi_pe: mark test as requiring multiple PEs"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add mpi marker to tests in TestDataTransfer, TestAtomicOperations, etc.
        if "TestDataTransfer" in str(item.nodeid) or \
           "TestAtomicOperations" in str(item.nodeid) or \
           "TestCollectiveOperations" in str(item.nodeid):
            item.add_marker(pytest.mark.mpi)
            item.add_marker(pytest.mark.multi_pe)
