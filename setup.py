"""
Setup configuration for OpenGovCorpus
"""
from pathlib import Path
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="opengovcorpus",
    version="0.1.0",
    author="Prajun Trital",
    author_email="prajuncs@gmail.com",
    description="A library for creating structured datasets and RAG embeddings from government websites",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prajun7/open-gov-corpus-py-library",
    project_urls={
        "Bug Reports": "https://github.com/prajun7/open-gov-corpus-py-library/issues",
        "Source": "https://github.com/prajun7/open-gov-corpus-py-library",
        "Documentation": "https://github.com/prajun7/open-gov-corpus-py-library#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*", "initial_scraping_notebook_code"]),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.32",
        "beautifulsoup4>=4.14",
        "pandas>=2.3",
        "openai>=2.7",
        "chromadb>=1.3",
        "tqdm>=4.67",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "government",
        "scraping",
        "dataset",
        "rag",
        "embeddings",
        "vector-database",
        "nlp",
        "machine-learning",
        "open-data",
        "civic-tech",
    ],
    include_package_data=True,
    zip_safe=False,
)

