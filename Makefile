.PHONY: format style test paper docs


format:
	@printf "Automatically formatting code...\n"
	uv run isort .
	uv run black .
	@printf "\033[1;34mAuto-formatting complete!\033[0m\n\n"

style:
	@printf "Checking code style...\n"
	uv run black --check --diff --config pyproject.toml --verbose .
	@printf "\033[1;34mCode style checks pass!\033[0m\n\n"

test:  # Run the test suite.
	uv run pytest \
		-v .\
		--cov=./jax_unirep \
		--cov-report term-missing

paper:
	cd paper && bash build.sh

docs:
	mkdocs build
