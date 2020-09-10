.PHONY: format style test paper


format:
	@printf "Checking code style with black...\n"
	isort .
	black .
	@printf "\033[1;34mFormatting complete!\033[0m\n\n"

style:
	@printf "Checking code style...\n"
	black -l 79 . --check
	@printf "\033[1;34mCode style checks pass!\033[0m\n\n"

test:  # Test code using pytest.
	pytest -v . --cov=./jax_unirep --cov-report term-missing

paper:
	cd paper && bash build.sh
