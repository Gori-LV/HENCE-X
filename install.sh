#!/bin/bash
conda create -y -n hence_x python=3.8
source activate hence_x
conda install -y pytorch==1.8.0 -c pytorch
pip install scipy
CUDA="cu102"
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
pip install torch-geometric
pip install tqdm
pip install pgmpy==0.1.17
pip install networkx==2.5
pip install pandas==1.2.3