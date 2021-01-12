.PHONY: format style test paper docs


format:
	@printf "Automatically formatting code...\n"
	isort -rc .
	black .
	@printf "\033[1;34mAuto-formatting complete!\033[0m\n\n"

style:
	@printf "Checking code style...\n"
	black --check --diff --config pyproject.toml --verbose .
	@printf "\033[1;34mCode style checks pass!\033[0m\n\n"

fasttest:  # Run fast tests using pytest.
	pytest \
		-m "not slow" \
		-v .\
		--cov=./jax_unirep \
		--cov-report term-missing

slowtest:  # Run fast tests using pytest.
	pytest \
		-m "slow" \
		-v .\
		--cov=./jax_unirep \
		--cov-report term-missing

paper:
	cd paper && bash build.sh

docs:
	mkdocs build
