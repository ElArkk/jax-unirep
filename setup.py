from setuptools import find_packages, setup

setup(
    name="jax_unirep",
    version="0.1",
    packages=find_packages(),
    package_data={
        '': ["LICENSE", "weights/*"],
    },
    include_package_data=True,
)
