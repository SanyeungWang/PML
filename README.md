# PML: Progressive Margin Loss for Long-tailed Age Classification
This repository is the official implementation of paper: "PML: Progressive Margin Loss for Long-tailed Age Classification"(CVPR 2021)
[[Paper(CVF)]](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_PML_Progressive_Margin_Loss_for_Long-Tailed_Age_Classification_CVPR_2021_paper.html)
[[Paper(arXiv)]](https://arxiv.org/abs/2103.02140)

<div align="center">
    <img src=./flowchart.png>
</div>

## Datasets
[[_Morph II_]](https://paperswithcode.com/dataset/morph) |
[[_FG-NET_]](https://paperswithcode.com/dataset/fg-net) |
[[_ChaLearn LAP 2015_]](http://chalearnlap.cvc.uab.es/dataset/18/description/) |
[[_IMDB-WIKI_]](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

## Training
We use (SNMC) Single Node Multi-GPU Cards training (with DistributedDataParallel) to get better performance.
```
 python -m torch.distributed.launch --nproc_per_node=2  --master_port 29502 ./train.py  --config ./configs/chalearn/exp_margin.yml
```

## Testing
We test while training to save the best model.

__Update(2022.10.05): Add more test files__  
Different test methods will lead to different test results. For example, the result of batch test is better than that of single image test. The test_save_npz_plot.py can achieve the lowest MAE.
```
ages = torch.sum(outputs * torch.Tensor([i for i in range(101)]).cuda(), dim=1)
error = abs(ages - age).cpu().clone().detach().numpy()
print('current mae {}'.format(np.mean(error)))
```

__If you have any problems, please feel free to contact us on our email or open an issue.__

## Experiments
<div align="center">
    <img src=./Curves_Tables/1.png width="400">
    <img src=./Curves_Tables/2.png width="400">
    <img src=./Curves_Tables/3.png width="400">
    <img src=./Curves_Tables/4.png width="400">
    <img src=./Curves_Tables/5.png width="800">
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
