# Deep Learning-Enhanced Non-Invasive Detection of Pulmonary Hypertension and Subtypes via Chest Radiographs, Validated by Catheterization

- [Introduction](#introduction)
- [Usage](#usage)
- [Acknowledgement](#acknowledgement)
- [References](#references)

## Introduction
This reposity contains the code for the paper "Deep Learning-Enhanced Non-Invasive Detection of Pulmonary Hypertension and Subtypes via Chest Radiographs, Validated by Catheterization".
## Usage
### 1. Install
Install requirements, with Python 3.8 or higher, using pip3.

```bash
pip3 install -r requirements.txt
pip3 install -r requirements-dev.txt
```

### 2. Prepare the dataset
We have converted them into txt format,Extract them under {ROOT}/data, your directory tree should look like this:

```
${ROOT}/data
├── imgs
|   |—— 0.dcm
└── |—— 1.dcm
    |—— ...   
|—— train.txt
|—— val.txt
|—— test.txt

```
Please note that the dicom images and labels are indicative and not used in the paper.

### 3. Train
#### Sample single-process running code:
```bash
python train.py /CXR-PH-Net/data \
    --train_file train.txt \
    --val_file val.txt \
    --model vgg16 \
    --epochs 500 \
    --batch-size 24 \
    --validation-batch-size 24 \
    --lr 0.001 \
    --sched cosine \
    --num-classes 2 \
    --input-size 3 1024 1024 \
    --output /CXR-PH-Net/results/ \
```
### 4. Inference
We can run the following code, and the model prediction results will be saved in JSON format.
```bash
python inference.py \
    --data-dir /path/to/test/ \
    --model-name vgg16 \
    --num-classes 2 \
    --model-path /path/to/modelcheckpoint/*.pt \
    --use-gpu True \
    --save-path /CXR-PH-Net/results/test
```


## Acknowledgement
Our codebase heavily relies on [huggingface](https://github.com/huggingface/pytorch-image-models). Please check out their repo for more information, and consider citing them in addition to our manuscript if you use this codebase.
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
## References
- [VGG](https://arxiv.org/abs/1409.1556)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [HRNet](https://arxiv.org/abs/1908.07919)
- [ViT](https://arxiv.org/abs/2010.11929)