from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prometheum",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of Prometheum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/prometheum",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Add your package dependencies here
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "flake8>=3.9",
            "black>=21.5b2",
            "mypy>=0.812",
        ],
    },
)

