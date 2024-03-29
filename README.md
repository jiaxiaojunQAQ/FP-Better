<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Fast Propagation is Better: Accelerating Single-Step </br> Adversarial Training via Sampling Subnetworks (TIFS2024)</h1>
<p align='left' style="text-align:left;font-size:1.2em;">
<b>
    [<a href="https://ieeexplore.ieee.org/document/10471619" target="_blank" style="text-decoration: none;">Project Page</a>] |
    [<a href="https://arxiv.org/pdf/2310.15444.pdf" target="_blank" style="text-decoration: none;">arXiv</a>] &nbsp;
</b>
</p>

## Introduction
![Adversarial example generation of the proposed FGSM-SDI](/imgs/FP_Better.png)
<p align="center">
Overview of our FP-Better. 
</p>


> In this work, we propose to exploit the interior building blocks of the model to improve efficiency. Specifically, we propose to dynamically sample lightweight subnetworks as a surrogate model during training. By doing this, both the forward and backward passes can be accelerated for efficient adversarial training. Besides, we provide theoretical analysis to show the model robustness can be improved by the single-step adversarial training with sampled subnetworks. Furthermore, we propose a novel sampling strategy where the sampling varies from layer to layer and from iteration to iteration. Compared with previous methods, our method not only reduces the training cost but also achieves better model robustness. Evaluations on a series of popular datasets demonstrate the effectiveness of the proposed FB-Better.
## Requirements

- Platform: Linux
- Hardware: 3090
- pytorch, etc.

## Train
```
python3.6 FGSM_DSD_cifar10.py  --out_dir ./output/ --data-dir cifar-data
```

## Test
```
python3.6 test_cifar10.py --model_path model.pth --out_dir ./output/ --data-dir cifar-data
```

## Trained Models
> The Trained models can be downloaded from the [Baidu Cloud](https://pan.baidu.com/s/1MZeV6fsSr6zdX9VMXQGonw)(Extraction: 1234) or the [Google Drive](https://drive.google.com/drive/folders/13v1-Wpkwu5Qj4Pq9OAQV0BCu61iyjW2w?usp=sharing)
