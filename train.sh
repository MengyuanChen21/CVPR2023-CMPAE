#!/bin/bash

SEED=0

python main.py --mode train -g default -e cmpae_s${SEED} -w --seed ${SEED} --gpu 0

echo "Begin to select thresholds from the validation set."
echo "Note: In this stage, five processes will run in parallel to save time."
echo "Please wait for about 10 min."

python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 0 --start_class 0 &
python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 1 --start_class 5 &
python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 2 --start_class 10 &
python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 3 --start_class 15 &
python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 4 --start_class 20 &

wait

python main.py --mode test -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 0