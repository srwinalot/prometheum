from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prometheum",
    version="0.1.0",
    author="Franklin Butahe",
    author_email="franklinbutahe@example.com",
    description="A flexible and powerful framework for data processing and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srwinalot/prometheum",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",               # Core numerical computations
        "pandas>=1.3.0",               # Data structures and analysis
        "scipy>=1.7.0",                # Scientific computing
        "scikit-learn>=1.0.0",         # Machine learning utilities
        "pyyaml>=6.0",                 # Configuration management
        "joblib>=1.1.0",               # Parallel computing
        "tqdm>=4.62.0",                # Progress bars
        "sqlalchemy>=1.4.0",           # Database ORM
        "requests>=2.26.0",            # HTTP client for API access
        "pydantic>=1.9.0",             # Data validation
        "pyarrow>=6.0.0",              # Fast data serialization
        "dask>=2022.1.0",              # Parallel computing
        "click>=8.0.0",                # CLI utilities
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",         # Basic plotting
            "seaborn>=0.11.0",           # Statistical data visualization
            "plotly>=5.0.0",             # Interactive visualizations
            "bokeh>=2.4.0",              # Interactive web visualizations
            "holoviews>=1.14.0",         # High-level visualization
        ],
        "ml": [
            "xgboost>=1.5.0",            # Gradient boosting
            "lightgbm>=3.3.0",           # Gradient boosting framework
            "catboost>=1.0.0",           # Gradient boosting
            "statsmodels>=0.13.0",       # Statistical models
            "tensorflow>=2.8.0",         # Deep learning
            "pytorch>=1.10.0",           # Deep learning
        ],
        "io": [
            "openpyxl>=3.0.0",           # Excel file support
            "xlrd>=2.0.0",               # Excel reading
            "beautifulsoup4>=4.10.0",    # HTML parsing
            "lxml>=4.6.0",               # XML processing
            "fastparquet>=0.8.0",        # Parquet file format
            "sqlalchemy-utils>=0.38.0",  # SQLAlchemy utilities
        ],
    },
)

