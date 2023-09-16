# HENCE-X

This is the official implement of the paper HENCE-X: Toward Heterogeneity-agnostic Multi-level Explainability for Deep Graph Networks.

[//]: # (![our_work]&#40;/intro_eg.png&#41;)
<p align="center">
  <img src="https://github.com/Gori-LV/HENCE-X/blob/main/intro_eg.png" />
</p>

[//]: # ([On Explainability of Graph Neural Networks via Subgraph Explorations]&#40;https://arxiv.org/abs/2102.05152&#41;)


## Installation
* Clone the repository 
* Create the env and install the requirements

```shell script
$ git clone https://github.com/Gori-LV/HENCE-X
$ cd HENCE-X
$ source ./install.sh
```

## Usage
* Download the required [datasets](https://hkustconnect-my.sharepoint.com/:f:/g/personal/glvab_connect_ust_hk/EpM6pkwnocROhKFBgJBIrqMBcfT0EX81WQA0RwpvqN923g?e=tNKQIF) to `/data`
* Download the [checkpoints](https://hkustconnect-my.sharepoint.com/:f:/g/personal/glvab_connect_ust_hk/Eg1VmSOyXFpHjIMP_gwXhssBR1OToeP4i75LUBlcmVgRCA?e=netLrt) to `/checkpoints`
* Run the searching scripts with corresponding dataset.
```shell script
$ source ./scripts.sh
``` 
The hyper-parameters used for different datasets are shown in this script.


## Examples
Run `*.ipynb` files in Jupyter Notebook or Jupyter Lab.  We provide examples on how to use HENCE-X to explain individual instances, and show the semantic meanings of output explanation in DBLP, IMDB and MUTAG dataset, respectively.

## Citation
Feel free to use our code and keep up with our progress, we kindly request you to cite our work.

```
@article{lv2023hence,
  title={HENCE-X: Toward Heterogeneity-Agnostic Multi-Level Explainability for Deep Graph Networks},
  author={Lv, Ge and Zhang, Chen Jason and Chen, Lei},
  journal={Proceedings of the VLDB Endowment},
  volume={16},
  number={11},
  pages={2990--3003},
  year={2023},
  publisher={VLDB Endowment}
}
```