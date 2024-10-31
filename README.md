# KAN'T Forget

## Thesis Project
This project was completed as part of my MSc in Computer Science from the University of Edinburgh.

## Description
This is an attempt to explore the benefits of using KAN networks to prevent Catastrophic Forgetting in Continual Learning settings. This repository implements the original paper on KAN networks [https://arxiv.org/abs/2404.19756] and augments the original idea so that the network can perform better when dealing with more complex tasks and higher-dimensional data such as image classification and image segmentation.
## How to use this repository
### Cloning
```
git clone https://github.com/harris2001/KAN-T-Forget
```
### Environment Setup & Installation
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```
### Reproducing results
```
python3 main.py
```
