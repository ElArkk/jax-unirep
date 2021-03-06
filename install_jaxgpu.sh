conda env update -f environment.yml

PYTHON_VERSION=cp37
CUDA_VERSION=cuda101
PLATFORM=manylinux2010_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.55-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade 'jax<0.2.0'

jupyter labextension install @jupyter-widgets/jupyterlab-manager
