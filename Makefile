# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.pyc"
CODESPELL_DIRS ?= mne_kit_gui/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build dist

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

wheel:
	$(PYTHON) setup.py sdist bdist_wheel

wheel_quiet:
	$(PYTHON) setup.py -q sdist bdist_wheel

testing_data:
	@python -c "import mne; mne.datasets.testing.data_path(verbose=True);"

pytest: test

test: in
	rm -f .coverage
	$(PYTESTS) mne_kit_gui

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

codespell:  # running manually
	@codespell --builtin clear,rare,informal,names,usage -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell --builtin clear,rare,informal,names,usage -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne

check-manifest:
	check-manifest -q --ignore .circleci/config.yml,doc,logo,.DS_Store

check-readme: clean wheel_quiet
	twine check dist/*

pep:
	@$(MAKE) -k flake pydocstyle docstring codespell-error check-manifest nesting check-readme

docstyle: pydocstyle
