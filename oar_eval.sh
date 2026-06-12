#!/bin/bash

#OAR -q production
#OAR -l gpu=1,walltime=2:00:00
#OAR -O .logs_eval
#OAR -E .errors_eval

cd /home/tderrien/Dataset_builder_MEL
source .venv/bin/activate
python eval/run_eval.py --config configs/eval/qwen_contrastive_10_text.yaml
python eval/run_eval.py --config configs/eval/qwen_ranking_10_text.yaml
