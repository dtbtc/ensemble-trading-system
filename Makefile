.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                        #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data_processor/processor.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Run tests
test:
	pytest tests/

## Train models
train:
	$(PYTHON_INTERPRETER) main.py --mode train

## Run backtest
backtest:
	$(PYTHON_INTERPRETER) main.py --mode backtest

#################################################################################
# PROJECT RULES                                                                   #
#################################################################################

## Set up python interpreter environment
create_environment:
	conda create --name trading_env python=3.8
	@echo ">>> conda env created. Activate with:\nconda activate trading_env"

## Install project in dev mode
install:
	pip install -e . 