import os
import subprocess

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

cwd = Path(__file__).resolve().parent

package_name = "CodeC"
version = "1.0.0"



def get_extensions():
    ext_dirs = cwd / package_name 
    ext_modules = []

    # Add rANS module
    rans_lib_dir = cwd / package_name / "third_party/ryg_rans"
    rans_ext_dir = ext_dirs / "rans"

    extra_compile_args = ["-std=c++17"]
    if os.getenv("DEBUG_BUILD", None):
        extra_compile_args += ["-O0", "-g", "-UNDEBUG"]
    else:
        extra_compile_args += ["-O3"]
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}.ans",
            sources=[str(s) for s in rans_ext_dir.glob("*.cpp")],
            language="c++",
            include_dirs=[rans_lib_dir, rans_ext_dir],
            extra_compile_args=extra_compile_args,
        )
    )

    # Add ops
    ops_ext_dir = ext_dirs / "ops"
    ext_modules.append(
        Pybind11Extension(
            name=f"{package_name}._CXX",
            sources=[str(s) for s in ops_ext_dir.glob("*.cpp")],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    )

    return ext_modules





setup(
    name=package_name,
    version=version,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch>=1.7.1",
        "torchvision",
        "pytorch-msssim",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
)
