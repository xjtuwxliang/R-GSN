# R-GSN: The Relation-based Graph Similar Network for Heterogeneous Graph
This is the code for paper
- R-GSN: The Relation-based Graph Similar Network for Heterogeneous Graph
![RGSN-General-Paradigm](https://github.com/xjtuwxliang/R-GSN/blob/main/pics/RGSN-General-Paradigm.png)


## 1. Environments
Ubuntu16.04 + NVIDIA-1080TI \
Anaconda + python 3.7 
```text
numpy==1.19.4
scipy==1.5.4
ogb==1.2.3
texttable==1.6.3
torch==1.6.0+cu101
torchvision==0.7.0+cu101
torch-cluster==1.5.8
torch-geometric==1.6.1
torch-scatter==2.0.5
torch-sparse==0.6.8
torch-spline-conv==1.2.0
```

## 2. Usage

run R-GSN
```bash
python rgsn.py --device $Device --conv_name "rgsn"\
--test_batch_size 64  \
--Norm4 True \
--FDFT  True \
--use_attack True 
```

run baseline R-GCN
```base
python rgsn.py --device $Device --conv_name "rgcn"
```

## 3. Results
| model | params | valid | test|
| ------ | ------ | ------ | ----|
| R-GCN | 154,366,772 | 47.61±0.68 | 46.78±0.67 |
| R-GSN | 154,373,028 | 51.82±0.41 | 50.32±0.37 |
