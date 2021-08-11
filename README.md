[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-random-neural-network/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=graph-random-neural-network)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-random-neural-network/node-classification-on-pubmed-with-public)](https://paperswithcode.com/sota/node-classification-on-pubmed-with-public?p=graph-random-neural-network)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-random-neural-network/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=graph-random-neural-network)

# GRAND
This is the code of paper: Graph Random Neural Network for Semi-Supervised Learning on Graphs [[arxiv](https://arxiv.org/pdf/2005.11079.pdf)]

Please cite our paper if you think our work is helpful to you:

```
@inproceedings{feng2020grand,
  title={Graph Random Neural Network for Semi-Supervised Learning on Graphs},
  author={Feng, Wenzheng and Zhang, Jie and Dong, Yuxiao and Han, Yu and Luan, Huanbo and Xu, Qian and Yang, Qiang and Kharlamov, Evgeny and Tang, Jie},
  booktitle={NeurIPS'20},
  year={2020}
}
```

## Requirements
* Python 3.7.3
* Please install other pakeages by 
``` pip install -r requirements.txt```

## Usage Example
* Running one trial on Cora:
```sh run_cora.sh ```
* Running 100 trials with random initializations on Cora:
```sh run100_cora.sh ```
* Calculating the average accuracy of 100 trails on Cora:
```python result_100run.py cora ```

## Results

Our model achieves the following accuracies on Cora, CiteSeer and Pubmed with the public splits:

| Model name   |   Cora    |  CiteSeer |  Pubmed   |
| ------------ | --------- | --------- | --------- |
| GRAND        |   85.4%   |    75.4%  |   82.7%   |

## Running Environment 

The experimental results reported in paper are conducted on a single NVIDIA GeForce RTX 2080 Ti with CUDA 10.0, which might be slightly inconsistent with the results induced by other platforms.

## The AMiner-CS Dataset
The AMiner-CS dataset can be downloaded from [google drive](https://drive.google.com/file/d/1yG5BP0GJKoB2Q07Uqd1DuC2tMf4EZo4u/view?usp=sharing) or [baidu drive](https://pan.baidu.com/s/1QWsioe2hPTFWyoL3aF6jlQ) with password l0pe.
This dataset is extracted from [AMiner Citation Graph](https://www.aminer.cn/citation). Each node of the graph corresponds to a paper in computer science, and edges represent citation relations between papers. We use averaged [GLOVE-100](https://nlp.stanford.edu/projects/glove/) word vector of paper abstract as the node feature vector. These papers are manually categorized into 18 topics based on their publication venues. We use 20 samples per class for training, 30 samples per class for validation and the remaining nodes for test in our expeirments.
