from setuptools import setup, find_packages

setup(
    name="coolsearch",
    version="1.0",
    description="CoolSearch: blackbox function optimization",
    author="marcu",
    packages=find_packages(),
    install_requires=["numpy", "polars", "tqdm"],
)
