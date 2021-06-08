# Efficient  Water  Desalination  with  Graphene  NanoporesObtained using Artificial Intelligence

This is the Github repository for the paper *"Efficient  Water  Desalination  with  Graphene  NanoporesObtained using Artificial Intelligence"*. In this work, we propose a graphene nanopore optimization framework via the combination of DRL and CNN for efficient water desalination. The DRL agent controls the growth of nanopore, while the CNN is employed to predict the water flux and ion rejection of the nanoporous graphene membrane at a certain external pressure. Experiments show that our framework can design nanopore structures that are promising in energy-efficient water desalination.

## Prerequisites
---
- Windows, Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
---
### Installation
```
$ git clone https://github.com/yuyangw/Graphene-RL-DQN.git
$ cd Graphene-RL-DQN
$ conda env create --name graphene-rl --file env.yml
$ conda activate graphene-rl
```

### Train the CNN model
```
$ python cnn.py
```

### Train the nanopore design DRL agent
```
$ python main.py
```

## Results
---
Here we show nanopore evolution controlled by the DRL agent

<p float="left">
    <img src="figs/graphene_evolve.gif" width="240">
    <img src="figs/graphene_evolve2.gif" width="240">
</p>

## Data
---
The data that support the findings of this study are available from the corresponding author upon reasonable request.