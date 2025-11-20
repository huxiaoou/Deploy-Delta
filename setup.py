from setuptools import setup
from Cython.Build import cythonize

ext_modules = cythonize(
    [
        "src/math_tools.py",
    ],
    compiler_directives={
        "language_level": "3",
        "annotation_typing": False,
    },
)

setup(
    name="ModuleFactorAlg",
    version="1.0.0",
    ext_modules=ext_modules,
    author="huxiaoou",
)
