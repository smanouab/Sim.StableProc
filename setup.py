from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stablesde",
    version="0.1.0",
    author="Author",
    author_email="author@example.com",
    description="A package for simulating stochastic differential equations driven by stable processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/stablesde",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=6.0.0",
            "sphinx>=7.0.0",
        ],
    },
)