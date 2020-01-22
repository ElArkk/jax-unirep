from setuptools import find_packages, setup

setup(
    name="jax_unirep",
    version="0.1",
    packages=["jax_unirep"],
#     data_files=[("weights", ["weights/1900_weights/*.npy"])],
    package_data={
        'jax_unirep': ["weights/1900_weights/*.npy"],
    },
    include_package_data=True,
)
