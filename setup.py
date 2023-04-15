from pathlib import Path

from setuptools import find_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [line.strip() for line in file.readlines()]


def read_version(fname="experiment/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


style_packages = ["black==22.3.0", "flake8==3.9.2", "isort==5.10.1"]
test_packages = ["pytest==7.1.2"]

setup(
    name="interpolating-neural-networks",
    packages=find_packages(),
    version=read_version(),
    license="MIT",
    author="Akash Sonowal",
    author_email="work.akashsonowal@gmail.com",
    url="https://github.com/akashsonowal/interpolating-neural-networks/",
    description="Interpolating Neural Networks in Asset Pricing Data. Supports Distributed Training in TensorFlow.",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="readme.md",
    keywords=["double descent", "deep learning", "generalization", "asset pricing"],
    python_requires=">=3.8",
    install_requires=[required_packages],
    extras_require={
        "dev": test_packages + style_packages + ["pre-commit==2.19.0"],
        "test": test_packages,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
