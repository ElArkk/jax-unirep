.PHONY: format style test paper


format:
	@printf "Checking code style with black...\n"
	isort -rc -y .
	black -l 79 .
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style...\n"
	black -l 79 . --check
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

test:  # Test code using pytest.
	pytest -v . --cov=./ 

paper:
	cd paper && bash build.sh
