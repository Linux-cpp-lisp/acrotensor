from setuptools import setup

from pathlib import Path
import glob

import torch
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    BuildExtension
)

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None

dirs = [
    str(Path(__file__).absolute().parent.parent / d)
    for d in ["", "exec", "tensor", "util", "ops", "kernel", "python/csrc"]
]

extensions = [
    (CUDAExtension if WITH_CUDA else CppExtension)(
        '_acrotensor',  # + ("_cuda" if WITH_CUDA else ""),
        sum(
            # Catch both CUDA and C++ source
            (glob.glob(d + "/*.c[pu]") for d in dirs),
            start=[]
        ),
        include_dirs=dirs,
        extra_compile_args=(
            ["-O3"]  # "-Wno-sign-compare"
            + (["-lnvrtc", "-lcuda", "-DACRO_HAVE_CUDA"] if WITH_CUDA else [])
        ),
    )
]

setup(
    name="acrotensor",
    version="0.1.0",
    author="Linux-cpp-lisp",
    description="",
    license="MIT",
    license_files="../LICENSE",
    python_requires=">=3.6",
    install_requires=[],
    packages=["acrotensor"],
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension}
)
