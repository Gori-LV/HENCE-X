#!/bin/sh
## dblp dataset
#python3 pipeline.py --dataset dblp --min_sampls 1000 --significance 0.05  --p_perturb 0.52 --sigma 0.01 --k 10
#
#python3 pipeline.py --dataset dblp --min_sampls 1000 --significance 0.05  --p_perturb 0.49 --sigma 0.011 --k 15
#
#python3 pipeline.py --dataset dblp --min_sampls 1000 --significance 0.05  --p_perturb 0.5 --sigma 0.009 --k 20
#
#python3 pipeline.py --dataset dblp --min_sampls 1000 --significance 0.05  --p_perturb 0.5 --sigma 0.009 --k 25
#
#python3 pipeline.py --dataset dblp --min_sampls 1000 --significance 0.05  --p_perturb 0.5 --sigma 0.009 --k 30
#
## imdb dataset
#python3 pipeline.py --dataset imdb --min_samples 1000 --significance 0.05 --p_perturb 0.5 --sigma 0.011 --k 10
#
#python3 pipeline.py --dataset imdb --min_samples 1000 --significance 0.05 --p_perturb 0.5 --sigma 0.01 --k 15
#
#python3 pipeline.py --dataset imdb --min_samples 1000 --significance 0.05 --p_perturb 0.5 --sigma 0.01 --k 20
#
#python3 pipeline.py --dataset imdb --min_samples 1000 --significance 0.05 --p_perturb 0.5 --sigma 0.01 --k 25
#
#python3 pipeline.py --dataset imdb --min_samples 1000 --significance 0.05 --p_perturb 0.5 --sigma 0.011 --k 30
#
## mutag dataset
#python3 pipeline.py --dataset mutag --min_samples 1000 --significnce 0.05 --p_perturb 0.91 --sigma 0.09 --k 10
#
#python3 pipeline.py --dataset mutag --min_samples 1000 --significnce 0.05 --p_perturb 0.91 --sigma 0.09 --k 15
#
#python3 pipeline.py --dataset mutag --min_samples 1000 --significnce 0.05 --p_perturb 0.91 --sigma 0.09 --k 20
#
#python3 pipeline.py --dataset mutag --min_samples 1000 --significnce 0.05 --p_perturb 0.91 --sigma 0.09 --k 25

python3 pipeline.py --dataset mutag --min_samples 1000 --significnce 0.05 --p_perturb 0.91 --sigma 0.09 --k 30
