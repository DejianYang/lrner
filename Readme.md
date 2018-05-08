# LR-NER
This repository is the implemented code based on PyTorch of our paper **《Improving Low Resource Named Entity Recognition using Cross-lingual Knowledge Transfer》**,our approach achieved improvements on two low resource languages (including Dutch and Spanish) and Chinese OntoNotes 4.0 dataset.


* [LR-NER](#lr-ner)
    * [1 Requirement](#1-requirement)
    * [2 Installation](#2-installation)
        * [PyTorch](#pytorch)
        * [Dependencies](#dependencies)
    * [3 Usage](#3-usage)
    * [4 Dataset](#4-dataset)
    * [5 Reference](#5-reference)

## 1 Requirement
```
Python : 2.7
PyTorch : >=0.3.0
```

## 2 Installation
### PyTorch
This code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/).

### Dependencies
You can install dependencies like this :

```
pip install -r requirements.txt
```

## 3 Usage
The default configuration is in the file **demo.train.config** and **demo.decode.config**.You can modify the parameters as you want.

In ***training*** status: : `CUDA_VISIBLE_DEVICES=0 python main.py --config demo.train.config`

In ***decoding*** status : `python main.py --config demo.decode.config`


## 4 Dataset

| Language | Dataset | Link |
| --- | --- | --- |
| Dutch | CoNLL-2002 | https://github.com/synalp/NER/tree/master/corpus/ |
| Spanish | CoNLL-2002 | https://github.com/synalp/NER/tree/master/corpus/ |
| Chinese | Ontonotes 4.0 | https://catalog.ldc.upenn.edu/ldc2011t03 |

| Translation | Link |
| --- | --- |
| MUSE | https://github.com/facebookresearch/MUSE |

| Word embedding | Link |
| --- | --- |
| Glove(english) | https://nlp.stanford.edu/projects/glove/|

## 5 Reference
[NCRF++: An Open-source Neural Sequence Labeling Toolkit
](https://github.com/jiesutd/NCRFpp)

[NER-pytorch](https://github.com/ZhixiuYe/NER-pytorch)

