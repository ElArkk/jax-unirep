.PHONY: format style fasttest slowtest paper docs


format:
	@printf "Automatically formatting code...\n"
	uv run isort .
	uv run black .
	@printf "\033[1;34mAuto-formatting complete!\033[0m\n\n"

style:
	@printf "Checking code style...\n"
	uv run black --check --diff --config pyproject.toml --verbose .
	@printf "\033[1;34mCode style checks pass!\033[0m\n\n"

fasttest:  # Run fast tests using pytest.
	uv run pytest \
		-m "not slow" \
		-v .\
		--cov=./jax_unirep \
		--cov-report term-missing

slowtest:  # Run slow tests using pytest.
	uv run pytest \
		-m "slow" \
		-v .\
		--cov=./jax_unirep \
		--cov-report term-missing

paper:
	cd paper && bash build.sh

docs:
	mkdocs build
