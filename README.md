# PML: Progressive Margin Loss for Long-tailed Age Classification
This repository will be the official implementation of paper: "PML: Progressive Margin Loss for Long-tailed Age Classification"(CVPR 2021)
[[Paper(CVF)]](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_PML_Progressive_Margin_Loss_for_Long-Tailed_Age_Classification_CVPR_2021_paper.html)
[[Paper(arXiv)]](https://arxiv.org/abs/2103.02140)

We would love to share innovations, this work is on the agenda.

## Training
We use (SNMC) Single Node Multi-GPU Cards training (with DistributedDataParallel) to get better performance.
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 path_of_train.py --config path_of_exp_margin.yml
```

## Testing
We test while training to save best model.

## Experiment
<div align="center">
    <img src=./Curves_Tables/1.png width="750">
    <img src=./Curves_Tables/2.png width="750">
    <img src=./Curves_Tables/3.png width="750">
    <img src=./Curves_Tables/4.png width="750">
    <img src=./Curves_Tables/5.png width="750">
</div>

## Citation
If you found this code or our work useful, please cite our paper.
```
@InProceedings{Deng_2021_CVPR,
    author    = {Deng, Zongyong and Liu, Hao and Wang, Yaoxing and Wang, Chenyang and Yu, Zekuan and Sun, Xuehong},
    title     = {PML: Progressive Margin Loss for Long-Tailed Age Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {10503-10512}
}
```
