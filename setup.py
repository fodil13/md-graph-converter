from setuptools import setup, find_packages

setup(
    name="md-graph-converter",
    version="0.1.0",
    author="Fodil Azzaz, PhD",
    author_email="azzaz.fodil@gmail.com",
    description="Convert molecular dynamics frames into EquiformerV2-compatible graphs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fodil13/md-graph-converter",
    packages=find_packages(),  # Automatically finds mdgraph/
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "torch-geometric",
        "MDAnalysis"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "mdgraph-pipeline=mdgraph.pipeline:run_equiformerv2_pipeline",
        ],
    },
)
