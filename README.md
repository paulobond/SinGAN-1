# SinGAN extension: application to inpainting
[Original paper Arxiv](https://arxiv.org/pdf/1905.01164.pdf) 

Forks https://github.com/tamarott/SinGAN

This repo essentially adds a `recover.py` file which should be used for inpainting.


Example 
```
python recover.py --input_name birds.png --fake_input_name birds_bis.png --disc_loss 0.01 --use_mask True
```

`input_name` is the file name of the uncorrupted image to train the model on (we assume that we have such an image).

`fake_input_name` is the file name of another image which has the same multi-level patch distribution. If `use_mask=True`, 
a mask is added to produce a corrupted image. Optional parameters allow you to control the mask size and position.

