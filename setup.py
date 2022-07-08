import setuptools
from distutils.util import convert_path


with open("README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt") as f:
    install_requires = [pkg.strip() for pkg in f.readlines() if pkg.strip()]
version_dict = {}
ver_path = convert_path("tabeval/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), version_dict)


setuptools.setup(
    name="tabeval",
    version=version_dict["__version__"],
    author="Jan Luca van den Busch",
    description="Python eval() on key-value-based numerical data structures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlvdb/tabeval",
    packages=setuptools.find_packages(),
    install_requires=install_requires)
