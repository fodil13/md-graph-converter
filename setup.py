from setuptools import setup, find_packages

setup(
    name="md-graph-converter",
    version="0.1.0",
    author="Fodil Azzaz, PhD",
    author_email="azzaz.fodil@gmail.com",
    description="Convert MD simulation frames into EquiformerV2-compatible graphs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fodil13/md-graph-converter",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "MDAnalysis",
        "torch",
        "torch-geometric",
    ],
    python_requires=">=3.8",
)
