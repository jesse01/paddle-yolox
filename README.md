<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduce
YOLOX for Paddle 2.1， YOLOX's Paper [report on Arxiv](https://arxiv.org/abs/2107.08430).


<img src="assets/git_fig.png" width="1000" >

## Benchmark

#### Standard Models.
|Model |size |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |39.6      |9.8     |9.0 | 26.8 | [github](https://github.com/jesse01/paddle-yolox/blob/main/model/yolox_s.pdparams) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.4      |12.3     |25.3 |73.8| [github](https://github.com/jesse01/paddle-yolox/blob/main/model/yolox_m.pdparams)  |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |50.0  |14.5 |54.2| 155.6 | [github](https://github.com/jesse01/paddle-yolox/blob/main/model/yolox_l.pdparams) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640  |**51.2**      | 17.3 |99.1 |281.9 | [github](https://github.com/jesse01/paddle-yolox/blob/main/model/yolox_x.pdparams)  |

#### Light Models.
|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.3  | 0.91 |1.08 | [github](https://github.com/jesse01/paddle-yolox/blob/main/model/yolox_nano.pdparams)  |
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.8 | 5.06 |6.45 | [github](https://github.com/jesse01/paddle-yolox/blob/main/model/yolox_tiny.pdparams) |

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX.
```shell
git clone git@github.com:jesse01/paddle-yolox.git
```
Step2. Install [apex](https://github.com/NVIDIA/apex).

```shell
# skip this step if you don't want to train model.
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
Step3. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.30 --nms 0.45 --tsize 640 --save_result
```
or
```shell
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result
```
Demo for video:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result
```


</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python tools/train.py -n yolox-s -b 8 -o
                         yolox-m
                         yolox-l
                         yolox-x
```
* -m: paddle.distributed.launch, multiple gpu training
* -b: total batch size, the recommended number for -b is num-gpu * 8

**Multi GPU Training**

python -m paddle.distributed.launch tools/train.py   -n yolox-s -b 64 -o
                                                        yolox-m
                                                        yolox-l
                                                        yolox-x

When using -f, the above commands are equivalent to:

```shell
python tools/train.py -f exps/default/yolox-s.py  -b 64 -o
                         exps/default/yolox-m.py
                         exps/default/yolox-l.py
                         exps/default/yolox-x.py
```

</details>


<details>
<summary>Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 8 --conf 0.001 [--fuse]
                         yolox-m
                         yolox-l
                         yolox-x
```
* --fuse: fuse conv and bn
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python tools/eval.py -n  yolox-s -c yolox_s.pth -b 1 --conf 0.001 --fuse
                         yolox-m
                         yolox-l
                         yolox-x
```

</details>


## Third-party resources
* Original repo: [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)


