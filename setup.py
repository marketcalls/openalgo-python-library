from setuptools import setup, find_packages

setup(
    name="openalgo",
    version="1.0.21",
    author="Rajandran R",
    author_email="rajandran@openalgo.in",
    description="A Python library for interacting with OpenAlgo's trading APIs with high-performance technical indicators",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://openalgo.in",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.23.0",
        "pandas>=1.2.0",
        "websocket-client>=1.8.0",
        "numpy>=1.21.0",
        "numba>=0.54.0"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    keywords=[
        "trading",
        "algorithmic-trading",
        "finance",
        "websocket",
        "market-data",
        "real-time", 
        "stock-market",
        "api-wrapper",
        "openalgo",
        "market-data",
        "trading-api",
        "stock-trading",
        "technical-analysis",
        "indicators",
        "rsi",
        "macd",
        "sma",
        "ema",
        "bollinger-bands",
        "supertrend",
        "atr",
        "volume-analysis"
    ],
    project_urls={
        "Documentation": "https://docs.openalgo.in",
        "Source": "https://github.com/openalgo/openalgo-python",
        "Tracker": "https://github.com/openalgo/openalgo-python/issues",
    },
)
