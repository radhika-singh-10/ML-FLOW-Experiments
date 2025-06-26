# MLFLOW Experiments


## Steps to run the experiment
1. Create and activate the virtual environment
```
python3 -m venv dev-env
```
```
source dev-env/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```

3. a. Steps to run the MLFlow for traditional ML model experiments
```
python train.py --alpha==0.5 l1_ratio==0.5
```
3. b. Steps to run the MLFlow for LLM model experiments
```
python test.py
```
4. Deactivate virtual environment
```
deactivate dev-env
```
