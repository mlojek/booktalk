all_code = src

install:
	pip install -e .

build_wheel:
	pip install build wheel twine
	python -m build --wheel . --outdir dist/

clean:
	git clean -fdx

format:
	isort ${all_code} --profile black
	black ${all_code}
	pyprojectsort pyproject.toml

check: format
	black ${all_code} --check
	isort ${all_code} --check --profile black
	pylint ${all_code}
	pyprojectsort pyproject.toml --check
