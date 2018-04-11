<img src="imgs/D.png"></img>
</br></br></br>
# Pedestrain-Synthesis-GAN

<a href="https://arxiv.org/abs/1804.02047">Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond</a>

## Preparing
Prepare your data before training. The format of your data should follow the file in `datasets`.
## Training stage
```bash
python train.py --dataroot data_path --name model_name --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode aligned --use_spp --no_lsgan --norm batch
```

## Testing stage
```bash
python test.py --dataroot data_path --name model_name --model pix2pix --which_model_netG unet_256 --which_direction BtoA  --dataset_mode aligned --use_spp --no_lsgan --norm batch
```
## Vision
Run `python -m visdom.server` to see the training process.
</br>

<img src="imgs/compare_3line.png"></img>
<img src="imgs/compare_cityscapes_1.png"></img>
<img src="imgs/compare_Tsinghua_1.png"></img>

