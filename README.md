# Residual Local Feature Network

Our team (ByteESR) won the **first place** in Runtime Track (Main Track) and the **second place** in Overall Performance Track (Sub-Track 2) of [NTIRE 2022 Efficient Super-Resolution Challenge](https://data.vision.ee.ethz.ch/cvl/ntire22/). 

| <sub> model </sub> | <sub> Runtime[ms] </sub> | <sub> Params[M] </sub> | <sub> Flops[G] </sub> |  <sub> Acts[M] </sub> | <sub> GPU Mem[M] </sub> |
|  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  |
|  RLFN_ntire  | 27.11  |  0.317  | 19.70  |  80.05  | 377.91  |

## Open-Source
For commercial reasons, we don't release training code temporarily, please refer to [EDSR framework](https://github.com/sanghyun-son/EDSR-PyTorch) and our paper for details.
- [x]  Paper of our method [[arXiv]](https://arxiv.org/abs/2205.07514)
- [x]  Report of our performance [[NTIRE22 official report]](https://arxiv.org/abs/2205.05675)
- [x]  The pretrained model and test code in challenge.

## Testing
We modified the [official test code](https://github.com/ofsoundof/IMDN). To reproduce our result in the ESR challenge, please install PyTorch >= 1.5.0.

run `python test_demo.py` to generate image results.  
All test results will be saved in the folder `data/DIV2K_test_LR_results`


