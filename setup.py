from setuptools import setup, find_packages

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
  install_requires = [
  ],
  extra_requires = {"test": test_packages},
  classifiers = [
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3.8",
          "Topic :: Science/Engineering :: Artificial Intelligence",            
  ],
)
