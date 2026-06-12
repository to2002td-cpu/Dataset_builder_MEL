#!/bin/bash

#OAR -q production 
#OAR -l /host=1,walltime=36:00:00
#OAR -O .logs
#OAR -E .errors

cd /home/tderrien/Dataset_builder_MEL
source .venv/bin/activate
wikiambig scrape --config configs/scrape/default.yaml