"""
setup.py for rocshmem4py - Python bindings for ROCSHMEM

Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""

import os
import sys
from pathlib import Path
from typing import List
import subprocess
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            import subprocess
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the extension")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import subprocess
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # CMake arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Add ROCM_PATH if set
        if 'ROCM_PATH' in os.environ:
            cmake_args.append(f'-DROCM_PATH={os.environ["ROCM_PATH"]}')

        # Add ROCSHMEM_HOME if set
        if 'ROCSHMEM_HOME' in os.environ:
            cmake_args.append(f'-DROCSHMEM_HOME={os.environ["ROCSHMEM_HOME"]}')

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', '-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def get_version():
    """Get version from __init__.py"""
    init_file = Path(__file__).parent / 'rocshmem4py' / '__init__.py'
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"


def get_long_description():
    """Get long description from README if it exists"""
    readme = Path(__file__).parent / 'README.md'
    if readme.exists():
        with open(readme, 'r', encoding='utf-8') as f:
            return f.read()
    return "Python bindings for ROCSHMEM library"


setup(
    name='rocshmem4py',
    version=get_version(),
    author='Advanced Micro Devices, Inc.',
    author_email='',
    description='Python bindings for ROCm Shared Memory (ROCSHMEM) library',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/ROCm/rocSHMEM',
    packages=setuptools.find_packages(where='.', include=['rocshmem4py*']),
    ext_modules=[CMakeExtension('_rocshmem4py', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[],
    extras_require={
        'test': ['pytest>=6.0', 'numpy>=1.19', 'mpi4py>=3.0'],
        'mpi': ['mpi4py>=3.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Software Development :: Libraries',
        'Topic :: System :: Distributed Computing',
    ],
    keywords='rocm hip rocshmem pgas hpc parallel distributed',
    project_urls={
        'Bug Reports': 'https://github.com/ROCm/rocSHMEM/issues',
        'Documentation': 'https://github.com/ROCm/rocSHMEM',
        'Source': 'https://github.com/ROCm/rocSHMEM',
    },
)
