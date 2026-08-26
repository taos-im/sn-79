# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import re
import os
import codecs
from os import path
from io import open
from setuptools import setup, find_packages

def _parse_req(req):
    """Normalise one requirements.txt line; return None for skippable entries."""
    if not req or req.startswith("#") or req.startswith("-"):
        return None
    if req.startswith("git+") or "@" in req:
        m = re.search(r"(#egg=)([\w\-_]+)", req)
        return m.group(2) if m else None
    return req


def read_requirements_split(path, marker_substring="GenTRX"):
    """Split ``requirements.txt`` into (core, extras) at a marker comment.

    Lines before the first comment whose body contains ``marker_substring`` go
    into ``install_requires``; everything after goes into the ``gentrx`` extra.
    This keeps ``pip install -e .`` lean by default while preserving the
    full-fat install via ``pip install -e .[gentrx]``.
    """
    with open(path, "r") as f:
        lines = f.read().splitlines()
    core, extra = [], []
    bucket = core
    for line in lines:
        if line.startswith("#") and marker_substring in line:
            bucket = extra
            continue
        parsed = _parse_req(line)
        if parsed is not None:
            bucket.append(parsed)
    return core, extra


requirements, gentrx_requirements = read_requirements_split("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "taos/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

setup(
    name="taos",
    version=version_string,
    description="taos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/taos-im/sn-79",
    author="taos.im",
    packages=find_packages(),
    include_package_data=True,
    author_email="to@taos.im",
    license="MIT",
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={"gentrx": gentrx_requirements},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Simulation",
        "Topic :: Scientific/Engineering :: Finance",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
