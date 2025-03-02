# frugal

## Setup

### Configure python.
Run the folowing commands to setup the python environment.
- pyenv-setup should be already done since Day 1 of the project phase.
- NEW : make sure to edit the .env file to set the correct path to the project directory.

```bash
make pyenv-setup
make env-setup
make activate
```

### Install the dependencies.
```bash
# Run this command when you update the pyproject.toml
make install

# Install the dev and test dependencies
make dev-install
make test-install
```

### Working on the project
**Good practices** //
- See [this document on coding best practices](https://github.com/Anatole-DC/datascience_starter_project/blob/master/documentation/best_practices.md)
- See [this document on git and github workflow](https://github.com/Anatole-DC/datascience_starter_project/blob/master/documentation/git_github_workflow.md)

**BYOM - Bring Your Own Model**
Our job is to focus on designing models, and finding the best one. To optimize this:
- The common functions to all models are already implemented in the module `frugal` : data (loading, preprocessing) and evaluation. We can use them to build and run our models!! :)
- These functions are located in the `/frugal` directory.
- The models are located in the `/apps/models/` directory. The design of the model is done in the `main.py` file. So we'll have as many `main.py`as models we want to test and compare.
- To create a new model, just run the following command:

```bash
make new-model MODEL_NAME="model_name"
```

This will generate a new model inside the `models` directory, with a `main.py` and a `README.md`.

The `main.py` is generated from the template file `/templates/model_template.py`.
