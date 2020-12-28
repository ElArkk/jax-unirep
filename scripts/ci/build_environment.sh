conda install -c conda-forge mamba
conda env create -f environment.yml
conda activate jax-unirep || source activate jax-unirep
python -m pip install --no-deps .
conda install conda-pack
conda pack -n jax-unirep -o jax-unirep.tar.gz
