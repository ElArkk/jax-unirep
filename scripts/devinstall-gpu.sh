mamba env update -f environment.yml
conda activate jax-unirep
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
pip install -e .
