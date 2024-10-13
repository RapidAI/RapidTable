# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys

import setuptools

from setuptools.command.install import install

MODULE_NAME = "slanet_plus_table"



setuptools.setup(
    name=MODULE_NAME,
    version="0.0.2",
    platforms="Any",
    long_description="simplify paddlex slanet plus table use",
    long_description_content_type="text/markdown",
    description="Tools for parsing table structures based paddlepaddle.",
    author="jockerK",
    author_email="xinyijianggo@gmail.com",
    url="https://github.com/RapidAI/RapidTable",
    license="Apache-2.0",
    include_package_data=True,
    install_requires=[
        "paddlepaddle==3.0.0b0",
        "PyYAML>=6.0",
        "opencv_python>=4.5.1.48",
        "numpy>=1.21.6",
        "Pillow",
    ],
    packages=[
        MODULE_NAME,
        f"{MODULE_NAME}.models",
        f"{MODULE_NAME}.table_matcher",
        f"{MODULE_NAME}.table_structure",
    ],
    package_data={"": ["inference.pdiparams","inference.pdmodel"]},
    keywords=["ppstructure,table,rapidocr,rapid_table"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
)
