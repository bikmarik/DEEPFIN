from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "calculate_data",
        ["dataGenius/calculate_data.cpp"],  # Path to your C++ file
    ),
]

setup(
    name="deepfin",
    version="0.1.0",
    author="Madara",
    description="Financial Engine using C++ and Python",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)