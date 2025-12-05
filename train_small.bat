@echo off
REM Ensure you're in conda env RL before running this
python train_dqn.py --config configs\dqn_small.yaml --seed 123
