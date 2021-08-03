# Description


The Python implementation for the publication [“Quantifying Predictive Uncertainty in Medical Image Analysis with Deep Kernel Learning”](https://arxiv.org/abs/2106.00638) on [ICHI2021](https://ichi2021.institute4hi.org/) (the 9th IEEE International Conference on Healthcare Informatics) using PyTorch and GPyTorch packages. 

# Project structure

```bash
.
├── cae_ba.py   
├── cae_dl.py    
├── cae_utils.py
├── data_utils.py
├── densenet.py    # a modified densenet to enable dropout behaviour during the inference phase
├── gp_layer.py    # SVGP-based output prediction layers
├── LICENSE
├── logging_conf.py
├── model_utils.py
├── plot_utils.py
├── pml_ba.py    
├── pml_dl.py  
├── requirements.txt
├── run_exp_ba_cae.py
├── run_exp_ba_dropout.py
├── run_exp_ba_metric.py
├── run_exp_ba.py
├── run_exp_dl_cae.py
├── run_exp_dl_dropout.py
├── run_exp_dl_metric.py
├── run_exp_dl.py
├── run_exp_gp_ba_cae.py
├── run_exp_gp_ba_metric.py
├── run_exp_gp_ba.py
├── run_exp_gp_dl_all_cae.py
├── run_exp_gp_dl_all_metric.py
└── run_exp_gp_dl_all.py
```

# Usage 

* All `.py` files should be able to run with `python xxx.py` after installing the packages specified in `requirements.txt`.
* The `.py` scripts prefixed with `run_exp_` can be used to generate (similar) results in [Table I/II/III/IV](https://arxiv.org/abs/2106.00638).
    * Dataset has to be downloaded for [the RSNA Bone Age dataset](https://stanfordmedicine.app.box.com/s/4r1zwio6z6lrzk7zw3fro7ql5mnoupcv/folder/42459416739) and [the Deep Lesion dataset](https://nihcc.app.box.com/v/DeepLesion). 
    * Scripts with `…_ba_…` are for the experiments with the RSNA Bone Age dataset.
    * Scripts with `…_dl_…` are for the experiments with the Deep Lesion dataset.

# Note

The code is published to ensure the reproducibility in the machine learning community. If you find the code helpful, please consider citing

```bib
@article{wu2021quantifying,
  title={Quantifying Predictive Uncertainty in Medical Image Analysis with Deep Kernel Learning},
  author={Wu, Zhiliang and Yang, Yinchong and Gu, Jindong and Tresp, Volker},
  journal={arXiv preprint arXiv:2106.00638},
  year={2021}
```


# License 

The code has a MIT license, as found in the [LICENSE](./LICENSE) file.