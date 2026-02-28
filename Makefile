setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

download:
	. .venv/bin/activate && python src/download_data.py

train:
	. .venv/bin/activate && python src/train.py

eval:
	. .venv/bin/activate && python src/evaluate.py

importance:
	. .venv/bin/activate && python src/feature_importance.py