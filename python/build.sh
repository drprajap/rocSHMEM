#!/bin/bash
################################################################################
# Build script for rocshmem4py
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[rocshmem4py]${NC} $1"
}

print_error() {
    echo -e "${RED}[rocshmem4py ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[rocshmem4py WARNING]${NC} $1"
}

# Check for required environment variables
print_msg "Checking environment..."

if [ -z "$ROCM_PATH" ]; then
    print_warning "ROCM_PATH not set, using default: /opt/rocm"
    export ROCM_PATH=/opt/rocm
fi

if [ -z "$ROCSHMEM_HOME" ]; then
    print_error "ROCSHMEM_HOME environment variable is not set!"
    print_error "Please set it to your ROCSHMEM installation directory:"
    print_error "  export ROCSHMEM_HOME=/path/to/rocshmem/install"
    exit 1
fi

if [ ! -d "$ROCSHMEM_HOME" ]; then
    print_error "ROCSHMEM_HOME directory does not exist: $ROCSHMEM_HOME"
    exit 1
fi

print_msg "ROCM_PATH: $ROCM_PATH"
print_msg "ROCSHMEM_HOME: $ROCSHMEM_HOME"

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:$LD_LIBRARY_PATH"
print_msg "LD_LIBRARY_PATH updated"

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_msg "Python version: $PYTHON_VERSION"

# Check for required Python packages
print_msg "Checking Python dependencies..."
python3 -c "import pybind11" 2>/dev/null || {
    print_warning "pybind11 not found, installing..."
    pip install pybind11
}

python3 -c "import cmake" 2>/dev/null || {
    print_warning "cmake not found, installing..."
    pip install cmake
}

# Parse command line arguments
BUILD_TYPE="Release"
INSTALL_MODE="develop"
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --install)
            INSTALL_MODE="install"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug     Build in debug mode"
            echo "  --install   Install instead of develop mode"
            echo "  --clean     Clean build directory before building"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_msg "Cleaning build directory..."
    rm -rf build dist *.egg-info
fi

# Build
print_msg "Building rocshmem4py in $BUILD_TYPE mode..."
export CMAKE_BUILD_TYPE=$BUILD_TYPE

if [ "$INSTALL_MODE" = "install" ]; then
    print_msg "Installing rocshmem4py..."
    pip install .
else
    print_msg "Installing rocshmem4py in development mode..."
    pip install -e .
fi

# Verify installation
print_msg "Verifying installation..."
python3 -c "import rocshmem4py; print(f'rocshmem4py version: {rocshmem4py.__version__}')" || {
    print_error "Failed to import rocshmem4py!"
    print_error "Make sure LD_LIBRARY_PATH includes ROCSHMEM libraries:"
    print_error "  export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:\$LD_LIBRARY_PATH"
    exit 1
}

print_msg "${GREEN}Build completed successfully!${NC}"
print_msg ""
print_msg "To use rocshmem4py, make sure to set:"
print_msg "  export LD_LIBRARY_PATH=$ROCSHMEM_HOME/lib:$ROCM_PATH/lib:\$LD_LIBRARY_PATH"
print_msg ""
print_msg "Run examples:"
print_msg "  python3 examples/basic_example.py"
print_msg "  mpirun -np 4 python3 examples/put_get_example.py"
print_msg ""
print_msg "Run tests:"
print_msg "  pytest tests/ -v"
