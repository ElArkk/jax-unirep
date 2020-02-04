from setuptools import find_packages, setup

setup(
    name="jax_unirep",
    version="0.1",
    packages=["jax_unirep"],
    package_data={"jax_unirep": ["weights/1900_weights/*.npy"],},
    install_requires=["jax", "jaxlib", "numpy",],
    include_package_data=True,
)
