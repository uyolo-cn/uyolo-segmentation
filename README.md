# <div align="center">uyolo-segmentation</div>

<div align="center">
    <p>SOTA Real Time Semantic Segmentation Models in PyTorch</p>
    <a href="https://colab.research.google.com/github/uyolo-cn/uyolo-segmentation/blob/main/demo/tutorials/demo-inference-with-pytorch.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    </a>
</div>

# <img src="./docs/assets/introduction_ico.png" width="30"/> Introduction

`uyolo-segmentation` is an open source semantic segmentation toolbox based on [`PyTorch Lightning`](https://github.com/Lightning-AI/lightning).

# <img src="./docs/assets/usage_ico.png" width="30"/> Getting Started

## <img src="./docs/assets/install.png" width="25"/> Installation

### 1. Install from Source

Clone the `uyolo-segmentation` repo from Github.

```shell
git clone https://github.com/uyolo1314/uyolo-segmentation
```

Run the following command, install `uyolo-segmentation` from source. If you make modification to `uyolo-segmentation/uyoloseg`, it will be efficient without reinstallation.

```shell
cd uyolo-segmentation
pip install -r requirements.txt
pip install -v -e .
```

## <img src="./docs/assets/train_model.png" width="25"/> How to Train

`uyolo-segmentation` is now using pytorch lightning for training. For both single-GPU or multiple-GPUs, run:

```shell
config_file=configs/custom.yaml
python tools/train.py --cfg ${config_file}
```

## <img src="./docs/assets/deployment.png" width="25"/> Deployment
- [ ] TensorRT
- [ ] OpenVINO
- [ ] NCNN
- [ ] Ascend

# <img src="./docs/assets/license_ico.png" width="30"/> License

`uyolo-segmentation` is released under the [Apache 2.0 license](./LICENSE).

# <img src="./docs/assets/acknowledgement_ico.png" width="30"/> Acknowledgement

- https://github.com/PaddlePaddle/PaddleSeg
- https://github.com/open-mmlab/mmsegmentation
- https://github.com/sithu31296/semantic-segmentation
- https://github.com/RangiLyu/nanodet
- https://github.com/Lightning-AI/lightning
