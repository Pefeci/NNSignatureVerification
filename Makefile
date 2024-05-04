style:
	python -m isort --profile=black .
	python -m black .

check:
	python -m flake8 --ignore=E501,W503,E203,E402 .
	python -m mypy .