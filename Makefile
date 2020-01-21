format:
	@printf "Checking code style with black...\n"
	isort -rc -y .
	black -l 79 .
	@printf "\033[1;34mBlack passes!\033[0m\n\n"