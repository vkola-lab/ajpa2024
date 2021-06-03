# HistoGraphs

This repository is the official PyTorch implementation of the graph CNNs for Digital Pathology Analysis:

It is based on the code from [Graph Isomorphism Networks](https://github.com/weihua916/powerful-gnns). Many thanks!

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run
Unzip the dataset file
```
unzip dataset.zip
```

and run

```
python main.py
```
To learn hyper-parameters to be specified, please type
```
python main.py --help
```
