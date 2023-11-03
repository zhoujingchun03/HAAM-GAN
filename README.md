# HAAM-GAN

This Repo includes the testing codes of our HAAM-GAN. (PyTorch Version).

If you use our code, please cite our paper and hit the star at the top-right corner. Thanks!


# Requirement
```
Python 3.7, Pytorch 1.11.0.
```


# Testing
```
1. Download the code
2. Put your testing images in the "data/input" folder
3. Python test.py
4. Find the result in "data/ouput" folder
5. You can find all the pre-trained model in https://drive.google.com/drive/folders/1h4OI-DIY0vgrjM2QrQXAyV3041xN8aHr?usp=sharing
Note that the PSNR_SSIM_UIQM.py provide the metrics code adopted our paper.
```

```
The validation data are in the "data/input" folder (underwater images), "data/gt" folder (grount truth images).
```

# Bibtex

```
@article{HAAMGAN,
  title={Hierarchical attention aggregation with multi-resolution feature learning for GAN-based underwater image enhancement},
  author={Zhang, Dehuan and Wu, Chenyu and Zhou, Jingchun and Zhang, Weishi and Li, Chaolei and Lin, Zifan},
  journal={Engineering Applications of Artificial Intelligence},
  volume={125},
  pages={106743},
  year={2023},
  publisher={Elsevier}
}
```
#  License
The code is made available for academic research purpose only. This project is open sourced under MIT license.

# Contact
If you have any questions, please contact Jingchun Zhou at zhoujingchun03@qq.com.

