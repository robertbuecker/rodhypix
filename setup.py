"""Legacy setup.py for backwards compatibility."""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

import numpy as np


class build_ext(_build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception:
            if not getattr(ext, "optional", False):
                raise
            self.warn(f"Building optional extension {ext.name!r} failed; using Python fallback.")


extensions = [
    Extension(
        "rodhypix.ty6_cpp",
        ["rodhypix/ty6_cpp.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        optional=True,
    )
]


setup(ext_modules=extensions, cmdclass={"build_ext": build_ext})
