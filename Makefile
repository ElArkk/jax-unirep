.PHONY: format style test paper


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
		-v .\
		--durations=0
		--cov=./jax_unirep \
		--cov-report term-missing \
		-m "not slow"

slowtest:  # Run fast tests using pytest.
	pytest \
		-v .\
		--durations=0
		--cov=./jax_unirep \
		--cov-report term-missing \
		-m "slow"

paper:
	cd paper && bash build.sh
