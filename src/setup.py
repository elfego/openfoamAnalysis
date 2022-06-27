from setuptools import Extension, setup
from Cython.Build import cythonize


ext_modules = [
    Extension("linearEqn",
              sources=["linearEqn.pyx"]),
    Extension("quadraticEqn",
              sources=["quadraticEqn.pyx"],
              libraries=["m"]),
    Extension("cubicEqn",
              sources=["cubicEqn.pyx"],
              libraries=["m"]),
    Extension("linalg",
              sources=["linalg.pyx"])
]

setup(name=["linearEqn", "quadraticEqn", "cubicEqn", "linalg"],
      ext_modules=cythonize(ext_modules))

