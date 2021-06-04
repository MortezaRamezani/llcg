# REF: https://github.com/cython/cython/wiki/PackageHierarchy

import os
import sys
import numpy

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import customize_compiler

from Cython.Distutils import build_ext
# from distutils.command.build_ext import build_ext
# from Cython.Build import cythonize


# scan the directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


# generate an Extension object from its dotted name
def make_extension(ext_name):
    extPath = ext_name.replace(".", os.path.sep)+".pyx"
    return Extension(
        ext_name,
        [extPath],
        language='c++',
        include_dirs=[numpy.get_include(), "."],   # adding the '.' to include_dirs is CRUCIAL!!
        extra_compile_args=['-O3', '-Wall', '-fopenmp'],
        extra_link_args=['-g', '-fopenmp'],
        # libraries = ['',],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )

# remove annoying warning


class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)


# get the list of extensions
ext_names = scandir('extension')

# and build up the set of Extension objects
extensions = [make_extension(name) for name in ext_names]

# finally, we can pass all this to distutils
setup(
    name="extension",
    packages=['utils', 'sparse'],
    cmdclass={'build_ext': my_build_ext},
    ext_modules=extensions,
    #   ext_modules=cythonize(extensions),
)
