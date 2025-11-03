#!/bin/bash
################################################################################
# Test runner script for rocshmem4py
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_msg() {
    echo -e "${GREEN}[rocshmem4py tests]${NC} $1"
}

# Check if pytest is installed
if ! python3 -c "import pytest" 2>/dev/null; then
    print_msg "Installing pytest and test dependencies..."
    pip install pytest numpy
fi

# Parse arguments
NUM_PROCS=2
TEST_FILE=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -np)
            NUM_PROCS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -f|--file)
            TEST_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -np N           Number of MPI processes (default: 2)"
            echo "  -v, --verbose   Verbose output"
            echo "  -f, --file FILE Run specific test file"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            print_msg "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set test path
if [ -n "$TEST_FILE" ]; then
    TEST_PATH="tests/$TEST_FILE"
else
    TEST_PATH="tests/"
fi

print_msg "Running single-process tests..."
pytest $TEST_PATH $VERBOSE -k "not multi_pe"

# Check if mpirun is available for multi-PE tests
if command -v mpirun &> /dev/null; then
    print_msg "Running multi-PE tests with $NUM_PROCS processes..."
    export WORLD_SIZE=$NUM_PROCS
    mpirun -np $NUM_PROCS pytest $TEST_PATH $VERBOSE -k "multi_pe" || true
else
    print_msg "${YELLOW}mpirun not found, skipping multi-PE tests${NC}"
fi

print_msg "Test run completed!"
