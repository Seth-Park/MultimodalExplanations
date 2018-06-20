# Multimodal Explanations: Justifying Decisions and Pointing to the Evidence
This repository contains the code for the following paper:

* DH. Park, LA. Hendricks, Z. Akata, A. Rohrbach, B. Schiele, T. Darrell, M. Rohrbach, *Multimodal Explanations: Justifying Decisions and Pointing to the Evidence.* in CVPR, 2018. ([PDF](https://arxiv.org/pdf/1802.08129.pdf)) 

## Installation

1. Install Python 3.
2. Install Caffe.
- Compile the `feature/20160617_cb_softattention` branch of [our fork of Caffe](https://github.com/akirafukui/caffe/). This branch contains Yang Gao’s Compact Bilinear layer, Signed SquareRoot layer, and L2 Normalization Layer ([dedicated repo](https://github.com/gy20073/compact_bilinear_pooling), [paper](https://arxiv.org/abs/1511.06062)) released under the [BDD license](https://github.com/gy20073/compact_bilinear_pooling/blob/master/caffe-20160312/LICENSE_BDD), and Ronghang Hu’s Soft Attention layers ([paper](https://arxiv.org/abs/1511.03745)) released under BSD 2-clause.
3. Download this repository or clone with Git, and then enter the root directory of the repository:
`git clone https://github.com/Seth-Park/MultimodalExplanations.git && cd MultimodalExplanations`


## Datasets Download & Preprocess
### VQA-X
1. Download and setup the VQA v2.0 dataset by following the instructions and directory structure described [here](https://github.com/GT-Vision-Lab/VQA).
2. Download the VQA-X dataset from [this google drive link](https://drive.google.com/drive/u/0/folders/1Cr9JRXDmjks_wmi-a9eIe4SWSwWKcCk7).
3. After unzipping the datasets, symlink the `VQA` directory setup in step 1 as `MultimodalExplanation/PJ-X-VQA/VQA-X` and place the data accordingly so that the file sructure looks as the following:

```
MultimodalExplanation/PJ-X-VQA/VQA-X/
    Annotations/
        train_exp_anno.json
        val_exp_anno.json
        test_exp_anno.json
        v2_mscoco_train2014_annotations.json
        v2_mscoco_val2014_annotations.json
        visual/
            val/
            test/
    Images/
        train2014
        val2014
    Questions/
        v2_OpenEnded_mscoco_train2014_questions.json
        v2_OpenEnded_mscoco_val2014_questions.json
    Features/  # this will be the directory to which we extract visual features!
    ...
```
4. We use ResNet-152 model to extract the visual features. Download the ResNet-152 caffemodel from [here](https://github.com/KaimingHe/deep-residual-networks).
5. Modify the `config.py` file in the `proprocess` directory to specify source path and destination path (should be `PJ-X-VQA/VQA-X/Features`). Then, start extracting the visual features.
```
cd preprocess
# fix config.py file
python extract_resnet.py
cd ..
```

### ACT-X
1. Download images for MPII Human Pose Dataset, Version 1.0 from [here](http://human-pose.mpi-inf.mpg.de/#download).
2. Download the ACT-X dataset from [this google drive link](https://drive.google.com/drive/u/0/folders/1Cr9JRXDmjks_wmi-a9eIe4SWSwWKcCk7).
3. After unzipping the datasets, symlink it to PJ-X-ACT/ACT-X and place the data accordingly so that the file sructure looks as the following:
```
MultimodalExplanation/PJ-X-ACT/ACT-X
    textual/
        exp_train_split.json
        exp_val_split.json
        exp_test_split.json
    visual/
        val/
        test/
    Features/  # this will be the directory to which we extract visual features!
    ...
```
4. We use ResNet-152 model to extract the visual features. Download the ResNet-152 caffemodel from [here](https://github.com/KaimingHe/deep-residual-networks).
5. Modify the `config.py` file in the `proprocess` directory to specify source path and destination path (should be `PJ-X-ACT/ACT-X/Features`). Then, start extracting the visual features.
```
cd preprocess
# fix config.py file
python extract_resnet.py
cd ..
```

## Training

### VQA-X
1. We use pretrained VQA model (using VQA training set) for the explanation task. Download the pretrained VQA caffemodel from [here](https://drive.google.com/drive/u/0/folders/1zQ4I8GrALJhvOfdzdgKAMriAHqQjUKal).
2. In the `PJ-X-VQA/model` directory, you will see `vdict.json` and `adict.json`. These are json files for the vocabulary and answer candidates used in our pretrained VQA model. Loading the json files will give you python dictionaries that map a word/answer to index. It is important to use these key-value mappings when using our pretrained VQA model. If training from scratch or using a different model, you will have to provide your own `vdict.json` and `adict.json`.
2. Modify the `config.py` file in `PJ-X-VQA` directory (i.e. set the path to where the pretrained VQA caffemodel is) and then start training:
```
cd PJ-X-VQA
# fix config.py file
python train.py
```

### ACT-X
1. For activity classification we do not use a pretrained network, so `vdict.json` and `adict.json` will be automatically generated in `PJ-X-ACT/model` directory. 
2. Modify the `config.py` file in `PJ-X-ACT` directory and then start training:
```
cd PJ-X-ACT
# fix config.py file
python train.py
```


## Generating Explanations

### VQA-X
1. The model prototxt, answer dictionary, vocab dictionary, and explanation dictionary will all be stored in `PJ-X-VQA/model` directory after training.
2. Provide this directory as input to the following command:
```
cd PJ-X-VQA/generate_vqa_exp
python generate_explanation.py --ques_file ../VQA-X/Questions/v2_OpenEnded_mscoco_val2014_questions.json --ann_file ../VQA-X/Annotations/v2_mscoco_val2014_annotations.json --exp_file ../VQA-X/Annotations/val_exp_anno.json --gpu 0 --out_dir ../VQA-X/results --folder ../model/ --model_path $PATH_TO_CAFFEMODEL --use_gt --save_att_map
```
The command will save generated textual and visual explanations in the directory designated by `--our_dir` .

### ACT-X
1. Similar to VQA-X, run this command to start generating explanations:
```
cd PJ-X-ACT/generate_act_exp
python generate_explanation.py --ann_file ../ACT-X/textual/exp_val_split.json --gpu 0 --out_dir ../ACT-X/results --folder ../model --model_path $PATH_TO_CAFFEMODEL --use_gt --save_att_map
```
The command will save generated textual and visual explanations in the directory designated by `--our_dir` .
