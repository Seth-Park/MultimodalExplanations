# Multimodal Explanations: Justifying Decisions and Pointing to the Evidence
This repository contains the code for the following paper:

* DH. Park, LA. Hendricks, Z. Akata, A. Rohrbach, B. Schiele, T. Darrell, M. Rohrbach, *Multimodal Explanations: Justifying Decisions and Pointing to the Evidence.* in CVPR, 2018. ([PDF](https://arxiv.org/pdf/1802.08129.pdf)) 
```
```

## Installation

1. Install Python 3.
2. Install Caffe.
- Compile the `feature/20160617_cb_softattention` branch of [our fork of Caffe](https://github.com/akirafukui/caffe/). This branch contains Yang Gao’s Compact Bilinear layer, Signed SquareRoot layer, and L2 Normalization Layer ([dedicated repo](https://github.com/gy20073/compact_bilinear_pooling), [paper](https://arxiv.org/abs/1511.06062)) released under the [BDD license](https://github.com/gy20073/compact_bilinear_pooling/blob/master/caffe-20160312/LICENSE_BDD), and Ronghang Hu’s Soft Attention layers ([paper](https://arxiv.org/abs/1511.03745)) released under BSD 2-clause.
3. Download this repository or clone with Git, and then enter the root directory of the repository:
`git clone https://github.com/Seth-Park/MultimodalExplanations.git && cd MultimodalExplanations`
