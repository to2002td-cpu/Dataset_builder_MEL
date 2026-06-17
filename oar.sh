#!/bin/bash

#OAR -q production 
#OAR -l /host=1,walltime=2:00:00
#OAR -O .1.logs
#OAR -E .1.errors

cd /home/tderrien/Dataset_builder_MEL
source .venv/bin/activate
wikiambig scrape --config configs/scrape/default.yaml -s s6,s7