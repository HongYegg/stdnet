# Saptial and Temporal Difference Network for Real-time salient object detection

## Introduction
This paper addresses the challenge of deploying salient object detection (SOD) on resource-constrained devices with real-time performance. While recent advances in deep neural networks have improved SOD, existing top-leading models are computationally expensive. We propose an efficient network design that combines traditional wisdom on SOD and the representation power of modern CNNs. Like biologically-inspired classical SOD methods relying on computing contrast cues to determine saliency of image regions, our model leverages Pixel Difference Convolutions (PDCs) to encode the feature contrasts. Differently, PDCs are incorporated in a CNN architecture so that the valuable contrast cues are extracted from rich feature maps. For efficiency, we introduce a difference convolution reparameterization (DCR) strategy that embeds PDCs into standard convolutions, eliminating computation and parameters at inference. Additionally, we introduce SpatioTemporal Difference Convolution (STDC) for video SOD, enhancing the standard 3D convolution with spatiotemporal contrast capture. Our models, SDNet for image SOD and STDNet for video SOD, achieve significant improvements in efficiency-accuracy trade-offs. On a Jetson Orin device, our models with < 1M parameters operate at 46 FPS and 150 FPS on streamed images and videos, surpassing the second-best lightweight models in our experiments by more than 2× and 3× in speed with superior accuracy.

<div align=center>
<img src="https://github.com/user-attachments/assets/8726815b-2b4f-4097-8c96-f16f88936351" width="80%"><br>
</div>

Coding style is based on [Pixel Difference Convolution](https://github.com/zhuoinoulu/pidinet).

## Environment (which we develop with)
- Ubuntu 20.04 + cuda 11.7
- RTX 3090 x 2
- python 3.8, pytorch 1.12

*Other versions may also work~ :)*

## Dataset

#### create data folders
```bash
# please change the dir for ROOTDIR to where you want to store your data
ROOTDIR=/to/rootdir
mkdir ${ROOTDIR}/vsod
mkdir ${ROOTDIR}/sod
```

#### Download SOD datasets
- Training on [DUTS-TR](http://saliencydetection.net/duts/).
- Testing on [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [PASCAL-S](http://cbs.ic.gatech.edu/salobj/), [DUT-O](http://saliencydetection.net/dut-omron/), [SOD](https://www.elderlab.yorku.ca/resources/salient-objects-dataset-sod/), [HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html), [DUTS-TE](http://saliencydetection.net/duts/).

For each dataset, please put the RGB images in folder `images` and ground truth images in `GT` (create them if there are no such folders), root them at ${ROOTDIR}/sod.

#### Download VSOD datasets
- [DAVSOD and DAVIS](https://github.com/DengPingFan/DAVSOD)
- VOS: [link1, where we downloaded it](https://github.com/Roudgers/DCFNet), or [link2](http://cvteam.net/projects/TIP18-VOS/VOS.html) (but please reorganize the files in its directory after downloading according to dataset/{train,val,test}\_video\_lst.txt)

Remember to unzip/unrar them to ${ROOTDIR}/vsod, and change the folder name for DAVSOD with the following scripts:
```bash
mv ${ROOTDIR}/vsod/Training\ Set ${ROOTDIR}/vsod/DAVSOD_Training_Set
mv ${ROOTDIR}/vsod/Validation\ Set ${ROOTDIR}/vsod/DAVSOD_Validation_Set
mv ${ROOTDIR}/vsod/Easy-35 ${ROOTDIR}/vsod/DAVSOD_Test_Set_Easy_35
```


## Evaluation

- Checkpoints of trained models can be found in [checkpoints](checkpoints).
- Saliency maps or our models can be downloaded at [saliency maps](https://github.com/isl-org/stdc-net/releases/download/v1.0/saliencyMaps.zip).

#### SOD (SDNet w/o Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='ECSSD+PASCAL-S+SOD+DUT-O+HKU-IS+DUTS-TE'
#testdata='ECSSD' # if you want to evaluate on only a single dataset
size=320
exp='sdnet_from_scratch'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_sod.py --model sdnet --inference-config baseline --train-config sdnet -j 4 --gpu 0 --datadir ${ROOTDIR}/sod --testdata ${testdata} --savedir results/$exp --evaluate checkpoints/${exp}.pth --size ${size}

# calcualting metrics finally.

cd Evaluation
bash eval_sod.sh ${testdata} ${exp} ${ROOTDIR}/sod
```

#### SOD (SDNet-A w/ Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='ECSSD+PASCAL-S+SOD+DUT-O+HKU-IS+DUTS-TE'
#testdata='ECSSD' # if you want to evaluate on only a single dataset
size=384
exp='sdneta_from_pretrained'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_sod.py --model sdneta --bn --inference-config baseline --train-config sdnet-a -j 4 --gpu 0 --datadir ${ROOTDIR}/sod --testdata ${testdata} --savedir results/$exp --evaluate checkpoints/${exp}.pth --size ${size}

# calcualting metrics finally.

cd Evaluation
bash eval_sod.sh ${testdata} ${exp} ${ROOTDIR}/sod
```

#### VSOD (STDNet w/o Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='DAVSOD'
#testdata='VOS'
#testdata='DAVIS'
size=256
exp='stdnet_from_scratch'
pos=_${testdata}
tid='cv cd ad'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_vsod.py --model stdnet --inference-config baseline --train-config sdnet -j 4 --gpu 0 --datadir ${ROOTDIR} --testdata ${testdata} --savedir results/$exp --size $size --evaluate checkpoints/${exp}${pos}.pth --tid ${tid}

# calcualting metrics finally.

cd Evaluation
bash eval_vsod.sh ${testdata} ${exp} ${ROOTDIR}
```

#### VSOD (STDNet-A w/ Imagenet pretraining)
```bash
ROOTDIR=/to/rootdir
testdata='DAVSOD'
#testdata='VOS'
#testdata='DAVIS'
size=256
exp='stdneta_from_pretrained'
pos=_${testdata}
tid='cv cd ad'

# Difference Convolution Reparameterization (DCR) first,
# generating saliency maps second,

python generate_salmaps_vsod.py --model stdneta --inference-config baseline --train-config sdnet-a -j 4 --gpu 0 --datadir ${ROOTDIR} --testdata ${testdata} --savedir results/$exp --size $size --evaluate checkpoints/${exp}${pos}.pth --tid ${tid}

# calcualting metrics finally.

cd Evaluation
bash eval_vsod.sh ${testdata} ${exp} ${ROOTDIR}
```

Note: For calculating metrics for VSOD models, the [matlab evaluation tool](https://github.com/DengPingFan/DAVSOD/tree/master/EvaluateTool) can also be used. We implement it with python in above scripts which support GPU and ouput the same results much faster.



## Training

```bash
ROOTDIR=/to/rootdir

# Train SDNet, w/o ImageNet pretraining
python train_sod.py --model sdnet --config sdnet --resume --gpu 0,1 --datadir ${ROOTDIR}/sod --savedir results/exp1

# Train SDNetA, w/ ImageNet pretraining
python train_sod.py --model sdneta --config sdnet-a --resume --gpu 0,1 --datadir ${ROOTDIR}/sod --preload checkpoints/sdneta_imagenet_pretrained_backbone.pth --savedir results/exp2

# Train STDNet-A with suitable hyperparameters for DAVSOD, w/ ImageNet pretraining
python train_vsod.py --model stdneta --config sdnet-a --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.01 --preload checkpoints/sdneta_imagenet_pretrained_backbone.pth --savedir results/exp3

# Train STDNet-A with suitable hyperparameters for VOS/DAVIS, w/ ImageNet pretraining
python train_vsod.py --model stdneta --config sdnet-a --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.1 --preload checkpoints/sdneta_imagenet_pretrained_backbone.pth --savedir results/exp4

# Train STDNet with suitable hyperparameters for DAVSOD, w/o ImageNet pretraining (here, lr_reduce is 0.1, stage1 backbone can be obtained from SDNet trained on DUTS-TR)
python train_vsod.py --model stdnet --config sdnet --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.1 --preload checkpoints/sdnet_stage1_backbone.pth --savedir results/exp5

# Train STDNet with suitable hyperparameters for VOS/DAVIS, w/o ImageNet pretraining (here, lr_reduce is 0.01, stage1 backbone can be obtained from SDNet trained on DUTS-TR)
python train_vsod.py --model stdnet --config sdnet --resume --gpu 0,1 --datadir ${ROOTDIR} --lr-reduce 0.01 --preload checkpoints/sdnet_stage1_backbone.pth --savedir results/exp6
```

## Testing FPS

```bash
## Test speed of SDNet, SDNet-A
shape=320
python speed_sod.py --model sdnet --config baseline -j 1 --gpu 0 --size $shape
python speed_sod.py --model sdneta --config baseline -j 1 --gpu 0 --size $shape


## Test speed of STDNet, STDNet-A
shape=256
python speed_vsod.py --model stdnet --config baseline -j 1 --gpu 0 --size $shape
python speed_vsod.py --model stdneta --config baseline -j 1 --gpu 0 --size $shape
```

## Acknowledgement
Repositories by which the code writing is inspired:

- [PiDiNet](https://github.com/zhuoinoulu/pidinet)
- [Evaluation-on-salient-object-detection](https://github.com/Jun-Pu/Evaluation-on-salient-object-detection)
- [DCFNet](https://github.com/Roudgers/DCFNet)
- [DAVSOD](https://github.com/DengPingFan/DAVSOD)

Friendly colleagues at Intel Lab.

