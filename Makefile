style:
	python3 -m isort --profile=black .
	python3 -m black .

check:
	python3 -m flake8 --ignore=E501,W503,E203 .
	python3 -m mypy .