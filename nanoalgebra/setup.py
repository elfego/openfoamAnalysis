from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize


VERSION = '0.0.1'
DESCRIPTION = '3D-specific vector and matrix algebra package'
LONG_DESCRIPTION = """A small collection of tools for vectors and matrices in
3-D. Specifically, the module has tools for computing eigenvalues and 
eigenvectors and the matrix invariants and a few additional tools for handling
symmetric and traceless matrices."""


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
    Extension("invars",
              sources=["invars.pyx"],
              libraries=["m"]),
    Extension("invarsh",
              sources=["invarsh.pyx"],
              libraries=["m"]),
    Extension("eig",
              sources=["eig.pyx"],
              libraries=["m"]),
    Extension("eigh",
              sources=["eigh.pyx"],
              libraries=["m"])
]

setup(
    name="nanoalgebra",
    version=VERSION,
    author="Elfego Ruiz Gutierrez",
    author_email="elfego.ruiz-gutierrez@newcastle.ac.uk",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=["nanoalgebra"],
    ext_modules=cythonize(extensions, language_level=3),
    install_requires=["cython", "numpy"]
)
