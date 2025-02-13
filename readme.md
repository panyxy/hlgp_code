# Hierarchical Learning-based Graph Partition for Large-scale Vehicle Routing Problems

## Introduction
This is the official PyTorch implementation for the paper titled ["Hierarchical Learning-based Graph Partition for Large-scale Vehicle Routing Problems"](https://arxiv.org/pdf/2502.08340).
The paper presents the HLGP framework, a new divide-and-conquer approach that combines global and local partition policies in graph partitioning tasks for subsequent node permutation. 
This method enables the scaling of neural solvers for the vehicle routing problem to handle over 10,000 nodes efficiently.
## Dependencies

Before executing the training process, please ensure that the necessary requirements have been installed.
```
conda create -n hlgp_env python=3.8
pip install torch==1.11.0+cu113 --index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Training

**RL-driven HLGP**
```
bash train_hlgp_rl.sh
```

**SL-driven HLGP**
```
bash train_hlgp_sl.sh
```


## Evaluation
We have provided the pretrained model for you to use directly.

**RL-driven HLGP**
```
bash eval_hlgp_rl.sh [batch_size] [problem_size] [gnn_topK] [eval_revision_iters] [dataset_name]

Arguments:
[problem_size]: 1000, 2000, 5000, 7000, 10000
[gnn_topK]: 100 (problem_size <=1000), 200 (problem_size >= 2000)
[eval_revision_iters]: 5
[dataset_name]: 1000, 2000, 5000, 7000, 10000, 1000_[explosion,gaussian,rotation], 2000_[gaussian,], 5000_[explosion,], 7000_[rotation,]
```

**SL-driven HLGP**
```
bash eval_hlgp_sl.sh [batch_size] [beam_size] [problem_size] [knns] [eval_revision_iters] [dataset_name]

Arguments:
[problem_size]: 1000, 2000, 5000, 7000, 10000
[beam_size]: 16 (problem_size=1000, 2000), 8 (problem_size=5000), 4 (problem_size=7000, 10000)
[knns]: 250 (problem=1000, 5000, 7000, 10000), 400 (problem_size=2000)
[eval_revision_iters]: 5
[dataset_name]: 1000, 2000, 5000, 7000, 10000, 1000_[explosion,gaussian,rotation], 2000_[gaussian,], 5000_[explosion,], 7000_[rotation,]
```

## Acknowledgement
We appreciate the anonymous reviewers, (S)ACs, and PCs of AAMAS2025 for their insightful
comments to further improve our paper and their service to the community.
We would like to thank the authors of GLOP ([implementation of GLOP](https://github.com/henry-yeh/GLOP))
and the authors of BQ ([implementation of BQ](https://github.com/naver/bq-nco)) for their invaluable contributions to the community.


## Citation

```
@inproceedings{
pan2025hlgp4cvrp,
title={Hierarchical Learning-based Graph Partition for Large-scale Vehicle Routing Problems},
author={Pan, Yuxin and Liu, Ruohong and Chen, Yize and Cao Zhiguang and Lin, Fangzhen},
booktitle={The 24th International Conference on Autonomous Agents and Multi-Agent Systems},
year={2025},
}
```

## License
This source code is provided solely for academic use. 
Please refrain from using it for commercial purposes without obtaining proper authorization from the author.

