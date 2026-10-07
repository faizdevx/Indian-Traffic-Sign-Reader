.PHONY: install test inspect prepare train-cnn train-resnet evaluate serve smoke

install:
	pip install -r requirements.txt

test:
	python -m pytest

inspect:
	python scripts/inspect_dataset.py --source $(SOURCE)

prepare:
	python scripts/prepare_dataset.py --source $(SOURCE)

train-cnn:
	python scripts/train_cnn.py

train-resnet:
	python scripts/train_resnet.py --mode finetune

evaluate:
	python scripts/evaluate_models.py

smoke:
	python scripts/train_cnn.py --smoke-test
	python scripts/train_resnet.py --mode frozen --smoke-test

serve:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
