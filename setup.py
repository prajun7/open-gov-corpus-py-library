"""
Setup configuration for OpenGovCorpus
"""
from setuptools import setup, find_packages

setup(
    name="opengovcorpus",
    version="0.1.0",
    author="Prajun Trital",
    description="A library for creating datasets and RAG embeddings from government websites",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.32",
        "beautifulsoup4>=4.14",
        "pandas>=2.3",
        "openai>=2.7",
        "chromadb>=1.3",
        "tqdm>=4.67",
    ],
)

