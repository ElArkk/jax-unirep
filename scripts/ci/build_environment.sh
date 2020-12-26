conda install -c conda-forge mamba
conda env create -f environment.yml
conda activate nams || source activate nams
python -m pip install --no-deps .
conda install conda-pack
conda pack -n jax-unirep -o jax-unirep.tar.gz
