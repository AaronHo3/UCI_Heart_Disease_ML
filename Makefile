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

train_gb:
	. .venv/bin/activate && python src/train_gb.py

compare:
	. .venv/bin/activate && python src/compare_models.py

loso:
	. .venv/bin/activate && python src/loso_validation.py

nested:
	. .venv/bin/activate && python src/nested_cv.py

calibrate:
	. .venv/bin/activate && python src/calibration_analysis.py

dca:
	. .venv/bin/activate && python src/decision_curve.py
