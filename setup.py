from pathlib import Path
from setuptools import setup, find_packages

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [line.strip() for line in file.readlines()]

style_packages = ["black==22.3.0", "flake8==3.9.2", "isort==5.10.1"]
test_packages = ["pytest==7.1.2"]

setup(
  name = 'interpolating-neural-networks',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license = 'MIT',
  author = 'Akash Sonowal',
  author_email = 'work.akashsonowal@gmail.com',
  url = 'https://github.com/akashsonowal/interpolating-neural-networks/',
  keywords = ["double descent", "deep learning", "generalization", "asset pricing"],
  install_requires = [required_packages],
  extra_requires = {"dev": style_packages + ["pre-commit==2.19.0"], 
                    "test": test_packages},
  classifiers = [
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3.8",
          "Topic :: Science/Engineering :: Artificial Intelligence",            
  ],
)
