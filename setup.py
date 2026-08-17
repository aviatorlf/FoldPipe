from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="foldpipe",
    version="0.3.0",
    author="FoldPipe Contributors",
    description="Bounded-memory asynchronous streaming for native PyTorch/PyG molecular-data shards.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aviatorlf/FoldPipe",
    packages=find_packages(include=["foldpipe", "foldpipe.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires='>=3.9',
    install_requires=[
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "biopython>=1.79",
        "google-api-python-client",
        "google-auth-oauthlib",
        "requests",
        "huggingface_hub",
    ],
    extras_require={
        "bench": ["nbformat>=5.0"],
        "dev": ["flake8", "nbformat>=5.0", "pytest", "pytest-mock"],
    },
)
