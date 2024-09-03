#!/usr/bin/python

from setuptools import setup, find_packages
import os
import sys

PACKAGE = "llm_chat"
NAME = "llmchat"
DESCRIPTION = "TRT-LLM Chat Server."
AUTHOR = "jun.zhang"
AUTHOR_EMAIL = "ewalker@live.cn"

TOPDIR = os.path.dirname(__file__) or "."
VERSION = __import__(PACKAGE).__version__

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package=find_packages(),
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    platforms="any",
    install_requires=["fastapi","uvicorn","colorama","openai","httpx"],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.10'

)
