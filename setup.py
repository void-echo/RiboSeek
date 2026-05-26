"""
setup.py — only here for the C extension. Metadata is in pyproject.toml.

The C extension `riboseek._nw_align` compiles nw_align.c into a shared
library that the Python wrapper loads via ctypes. We use distutils' Extension
mechanism so wheels built on each platform contain the right binary.
"""

import sys
from setuptools import setup, Extension


# Platform-specific compile flags.
# -O3 -march=native is fine for source distributions; for cibuildwheel /
# manylinux wheels we drop -march=native to keep portability.
extra_compile_args = ["-O3"]
extra_link_args = []

if sys.platform == "win32":
    # MSVC uses /O2 instead of -O3
    extra_compile_args = ["/O2"]
elif sys.platform == "darwin":
    extra_compile_args += ["-fPIC"]
else:
    extra_compile_args += ["-fPIC"]

nw_align_ext = Extension(
    name="riboseek._nw_align",
    sources=["src/riboseek/_native/nw_align.c"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=["m"] if sys.platform != "win32" else [],
)

setup(
    ext_modules=[nw_align_ext],
)
