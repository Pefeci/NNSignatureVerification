style:
	python -m isort --profile=black main.py model.py validator.py functions.py loader.py conf.py
	python -m black main.py model.py validator.py functions.py loader.py conf.py

check:
	python -m flake8 --ignore=E501,W503,E203,E402 main.py model.py validator.py functions.py loader.py conf.py
	python -m mypy main.py model.py validator.py functions.py loader.py conf.py