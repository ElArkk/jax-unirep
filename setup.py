from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("jax_unirep/version.py") as fp:
    version = fp.read().split('"')[1]

setup(
    name="jax_unirep",
    author="Arkadij Kummer, Eric J. Ma, Ivan Jayapurna",
    author_email="arkadij.kummer@gmail.com, ericmajinglong@gmail.com, ivanfj@berkeley.edu",
    description="A performant and user-friendly reimplementation of UniRep in JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ElArkk/jax-unirep",
    version=version,
    packages=["jax_unirep"],
    package_data={
        "jax_unirep": ["weights/uniref50/*/*.pkl"],
    },
    install_requires=[
        "jax",
        "jaxlib",
        "multipledispatch",
        "numpy",
        "optuna",
        "scikit-learn",
        "tqdm",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.6",
)
