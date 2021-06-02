from setuptools import setup
from torch.utils import cpp_extension
from pathlib import Path
import glob

dirs = [
    str(Path(__file__).absolute().parent.parent / d)
    for d in ["", "exec", "tensor", "util", "ops", "kernel", "python/csrc"]
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
    ext_modules=[cpp_extension.CppExtension(
        '_acrotensor',
        sum(
            (glob.glob(d + "/*.cpp") for d in dirs),
            start=[]
        ),
        include_dirs=dirs,
        extra_compile_args=["--std=c++11", "-g"],
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
