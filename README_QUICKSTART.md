
# Snake-RL v2 — Quickstart

## 1) Create/activate Conda env (Windows PowerShell)
conda create -n RL python=3.11 -y
conda activate RL
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## 2) Train
python train_dqn.py --config configs/dqn_small.yaml --seed 123

## 3) TensorBoard
tensorboard --logdir runs

## 4) CSV Summary
python make_report.py
