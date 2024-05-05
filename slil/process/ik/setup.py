from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(
        Extension("ik", ["ik.pyx"],
            extra_compile_args=["-Ox", "-Zi"],
            extra_link_args=["-debug:full"]),
        emit_linenums=True,
        gdb_debug=True),
    include_dirs=[np.get_include()]
    )