from setuptools import setup

setup(
    name='coolsearch',
    version='1.0',
    description='CoolSearch: blackbox function optimization',
    author='marcu',
    packages=['coolsearch'],  # same as name
    # external packages as dependencies
    install_requires=['numpy', 'polars'],
)
