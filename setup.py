from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys

extra = ["-O3", "-march=native"] if sys.platform != "win32" else ["/O2", "/std:c++14"]

ext = Extension(
    name="GTBenchmark.transform.gf_algos.algos",
    sources=["GTBenchmark/transform/gf_algos/algos.pyx"],
    language="c++",
    include_dirs=[np.get_include()],
    extra_compile_args=extra,
)

setup(
    name="GTBenchmark",
    version="0.0.1",
    packages=find_packages(),          # 自动发现包
    ext_modules=cythonize([ext], language_level=3,
                          compiler_directives=dict(
                              boundscheck=False, wraparound=False,
                              initializedcheck=False, cdivision=True)),
    zip_safe=False,
)
