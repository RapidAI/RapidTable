# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion


def get_readme():
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / "docs" / "doc_whl_rapid_table.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


MODULE_NAME = "rapid_table"
obtainer = GetPyPiLatestVersion()
latest_version = obtainer(MODULE_NAME)
VERSION_NUM = obtainer.version_add_one(latest_version)

if len(sys.argv) > 2:
    match_str = " ".join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    description="Tools for parsing table structures based ONNXRuntime.",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidStructure",
    license="Apache-2.0",
    include_package_data=True,
    install_requires=[
        "onnxruntime>=1.7.0",
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
    package_data={"": ["slanet-plus.onnx"]},
    keywords=["ppstructure,table,rapidocr,rapid_table"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.6,<3.13",
    entry_points={"console_scripts": [f"{MODULE_NAME}={MODULE_NAME}.main:main"]},
    extras_require={"torch": ["torch", "torchvision", "tokenizers"]},
)
