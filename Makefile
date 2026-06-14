setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

download:
	. .venv/bin/activate && python src/download_data.py

loso:
	. .venv/bin/activate && python src/loso_validation.py

nested:
	. .venv/bin/activate && python src/nested_cv.py

calibrate:
	. .venv/bin/activate && python src/calibration_analysis.py

dca:
	. .venv/bin/activate && python src/decision_curve.py

missingness:
	. .venv/bin/activate && python src/missingness.py

interpret:
	. .venv/bin/activate && python src/interpretability.py

fairness:
	. .venv/bin/activate && python src/fairness.py

conformal:
	. .venv/bin/activate && python src/conformal.py

manifest:
	. .venv/bin/activate && python src/manifest.py

test:
	. .venv/bin/activate && pytest -q

study: loso nested calibrate dca interpret missingness fairness conformal manifest
