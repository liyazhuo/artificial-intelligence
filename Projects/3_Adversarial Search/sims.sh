#!/bin/bash
python run_match.py -o MINIMAX -r 500 -t 500
python run_match.py -o MINIMAX -r 500 -t 350
python run_match.py -o MINIMAX -r 500 -t 250
python run_match.py -o MINIMAX -r 500 -t 150

python run_match.py -o GREEDY -r 500 -t 500
python run_match.py -o GREEDY -r 500 -t 350
python run_match.py -o GREEDY -r 500 -t 250
python run_match.py -o GREEDY -r 500 -t 150

python run_match.py -o RANDOM -r 500 -t 500
python run_match.py -o RANDOM -r 500 -t 350
python run_match.py -o RANDOM -r 500 -t 250
python run_match.py -o RANDOM -r 500 -t 150