from setuptools import Extension, setup
from Cython.Build import cythonize


extensions = [
    Extension("linearEqn",
              sources=["linearEqn.pyx"]),
    Extension("quadraticEqn",
              sources=["quadraticEqn.pyx"],
              libraries=["m"]),
    Extension("cubicEqn",
              sources=["cubicEqn.pyx"],
              libraries=["m"]),
    Extension("linalg",
              sources=["linalg.pyx"],
              libraries=["m"]),
    Extension("eig",
              sources=["eig.pyx"],
              libraries=["m"]),
    Extension("eigh",
              sources=["eigh.pyx"],
              libraries=["m"])
]


setup(
    name=[
        "linearEqn",
        "quadraticEqn",
        "cubicEqn",
        "linalg",
        "eig",
        "eigh"
    ],
    ext_modules=cythonize(
        extensions,
        language_level=3
    )
)
