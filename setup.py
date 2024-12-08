from setuptools import setup, find_packages

setup(
    name="openalgo",
    version="1.0.3",  # Updated version with data API features
    author="Rajandran R",
    author_email="rajandran@marketcalls.in",
    description="A Python library for interacting with OpenAlgo's trading APIs",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://openalgo.in",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas",  # Added pandas dependency for history data handling
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
