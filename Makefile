pyenv-setup:
	pyenv install 3.10.6
	pyenv virtualenv 3.10.6 frugal
	pyenv local frugal

env-setup:
	cp template.env .env

activate:
	pyenv activate frugal

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test-install:
	pip install -e ".[test]"

all-install:
	make install
	make dev-install
	make test-install

code-clean:
	black .
	flake8

tests:
	pytest

new-model:
	@mkdir -p apps/models/$(MODEL_NAME)
	@cp templates/model_template.py apps/models/$(MODEL_NAME)/main.py
	@touch apps/models/$(MODEL_NAME)/README.md
