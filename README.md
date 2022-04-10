# WildPASS
Panoramic Semantic Segmentation in the Wild

[**WildPASS Dataset**](https://drive.google.com/file/d/1DEIOpwdtWxdUBnvBXqe1KxzW-qez_KSY/view?usp=sharing)

[**WildPASS Dataset with City Names**](https://drive.google.com/file/d/1yHCdhe45IzDcHarYPKCdZCzwWR0PGdAF/view?usp=sharing)

[**WildPASS2K Dataset**](https://drive.google.com/file/d/1c9pQJe9OJcvW24rRg9pSTSgvCRklKuvv/view?usp=sharing)

The WildPASS dataset contains annotated 500 panoramas taken from 25 cities located on multiple continents for evaludation.

The WildPASS2K dataset contains 2000 unlabled panoramas taken from 40 cities, which can be used for facilitating domain adapation and creating pseudo labels.

For training, we suggest to use [**Mapillary Vistas**](https://drive.google.com/file/d/1c9pQJe9OJcvW24rRg9pSTSgvCRklKuvv/view?usp=sharing), or with a combination of [**IDD20K**](https://idd.insaan.iiit.ac.in/), [**Cityscapes**](https://www.cityscapes-dataset.com/), [**ApolloScape**](http://apolloscape.auto/scene.html), [**BDD10K**](https://bdd-data.berkeley.edu/), [**Audi A2D2**](https://www.a2d2.audi/a2d2/en.html), [**KITTI**](http://www.cvlibs.net/datasets/kitti/eval_semantics.php), [**KITTI-360**](http://www.cvlibs.net/datasets/kitti-360/), and [**WildDash2**](https://wilddash.cc/) datasets. 

![Example segmentation](figure_wildpass.jpg?raw=true "Example segmentation")

**Example**
```
CUDA_VISIBLE_DEVICES=0
python3 eval_color_fusion.py
--datasets 'MAP' 'IDD20K'
--is-fuse
--basedir /cvhci/data/
--subset val
--loadDir ../trained_models/
--loadWeights model_best.pth
--loadModel ecanet.py
--datadir /cvhci/data/WildPASS
```

## Publications
If you use our code or dataset, please consider citing any of the following papers:

**Capturing Omni-Range Context for Omnidirectional Segmentation.**
K. Yang, J. Zhang, S. Reiß, X. Hu, R. Stiefelhagen.
In IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), Nashville, TN, United States (Virtual), June 2021.
[[**PDF**](https://arxiv.org/pdf/2103.05687.pdf)]

```
@inproceedings{yang2021capturing,
title={Capturing Omni-Range Context for Omnidirectional Segmentation},
author={Yang, Kailun and Zhang, Jiaming and Rei{\ss}, Simon and Hu, Xinxin and Stiefelhagen, Rainer},
booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2021}
}
```

**Is Context-Aware CNN Ready for the Surroundings? Panoramic Semantic Segmentation in the Wild.**
K. Yang, X. Hu, R. Stiefelhagen.
IEEE Transactions on Image Processing (**TIP**), 2021.
[[**PDF**](http://www.yangkailun.com/publications/tip2021_kailun.pdf)]

```
@article{yang2021context,
title={Is Context-Aware CNN Ready for the Surroundings? Panoramic Semantic Segmentation in the Wild},
author={Yang, Kailun and Hu, Xinxin and Stiefelhagen, Rainer},
journal={IEEE Transactions on Image Processing},
volume={30},
pages={1866--1881},
year={2021},
publisher={IEEE}
}
```

## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
