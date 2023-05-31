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
* Run the searching scripts with corresponding dataset, parameters are included in the `.sh` file.
```shell script
$ source ./scripts.sh
``` 
## Examples
Run `*.ipynb` files in Jupyter Notebook or Jupyter Lab.


[//]: # (## Reference)

[//]: # (If you make advantage of Gem in your research, please cite the following in your manuscript:)

[//]: # ()
[//]: # (```)

[//]: # (@inproceedings{)

[//]: # (    wanyu-icml21,)

[//]: # (    title="{Generative Causal Explanations for Graph Neural Networks}",)

[//]: # (    author={Lin, Wanyu and Lan, Hao and Li, Baochun},)

[//]: # (    booktitle={International Conference on Machine Learning},)

[//]: # (    year={2021},)

[//]: # (    url={https://arxiv.org/pdf/2104.06643.pdf},)

[//]: # (})

[//]: # (```)

[//]: # (```shell script)

[//]: # ($ cd HENCE-X)

[//]: # ($ source ./scripts.sh)

[//]: # (``` )

[//]: # (The hyper-parameters for different models and datasets are shown in this script.)

[//]: # (In addition, we also provide the saved searching result.)

[//]: # (If you want to reproduce, you can directly download the )

[//]: # ([result]&#40;https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ERxIONDcl8xKswisrsbHo2MBoEwPAjFruUzwsLpESwalxA?e=IuFanz&#41;)

[//]: # ( to `HENCE-X/result`)

[//]: # (Moreover, if you want to train a new model for these datasets, )

[//]: # (run the training scripts for corresponding dataset.)

[//]: # (```shell script)

[//]: # ($ cd HENCE-X)

[//]: # ($ source ./models/train_gnns.sh )

[//]: # (```)

[//]: # (## Citations)

[//]: # (If you use this code, please cite our papers.)

[//]: # ()
[//]: # (```)

[//]: # (@misc{yuan2021explainability,)

[//]: # (      title={On Explainability of Graph Neural Networks via Subgraph Explorations}, )

[//]: # (      author={Hao Yuan and Haiyang Yu and Jie Wang and Kang Li and Shuiwang Ji},)

[//]: # (      year={2021},)

[//]: # (      eprint={2102.05152},)

[//]: # (      archivePrefix={arXiv},)

[//]: # (      primaryClass={cs.LG})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (```)

[//]: # (@article{yuan2020explainability,)

[//]: # (  title={Explainability in Graph Neural Networks: A Taxonomic Survey},)

[//]: # (  author={Yuan, Hao and Yu, Haiyang and Gui, Shurui and Ji, Shuiwang},)

[//]: # (  journal={arXiv preprint arXiv:2012.15445},)

[//]: # (  year={2020})

[//]: # (})

[//]: # (```)
