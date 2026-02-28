setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

download:
	. .venv/bin/activate && python src/download_data.py

train:
	. .venv/bin/activate && python src/train_logreg.py

eval:
	. .venv/bin/activate && python src/evaluate.py

importance:
	. .venv/bin/activate && python src/feature_importance.py

train_rf:
	. .venv/bin/activate && python src/train_rf.py

compare:
	. .venv/bin/activate && python src/compare_models.py
