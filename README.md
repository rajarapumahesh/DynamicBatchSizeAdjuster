# DynamicBatchSizeAdjuster
Welcome to the official repository for the Dynamic Batch Size Adjuster (DBSA) algorithm. This repository contains the source code used to evaluate DBSA’s performance on the CIFAR-10 and MNIST datasets, comparing it against static batch size methods. The code is organized for easy access and replication of our experiments.

**Repository Overview**
The code is structured under the DBSA Codes directory, split into two main datasets: CIFAR-10 and MNIST. Each dataset folder contains subfolders for static and DBSA implementations across three model architectures: CNN, ResNet, and RNN. Below is the folder structure:


DBSA Codes/
├── CIFAR-10/
│   ├── Static/
│   │   ├── CIFAR CNN STATIC.py
│   │   ├── CIFAR RESNET STATIC.py
│   │   └── CIFAR RNN STATIC.py
│   └── DBSA/
│       ├── CIFAR CNN DBSA.py
│       ├── CIFAR RESNET DBSA.py
│       └── CIFAR RNN DBSA.py
└── MNIST/
    ├── Static/
    │   ├── MNIST CNN STATIC.py
    │   ├── MNIST RESNET STATIC.py
    │   └── MNIST RNN STATIC.py
    └── DBSA/
        ├── MNIST CNN DBSA.py
        ├── MNIST RESNET DBSA.py
        └── MNIST RNN DBSA.py

**Experimental Design:** Isolating Batch Size Effects
A key principle in our study is to understand how batch size alone affects training performance, without the influence of different model architectures. To achieve this, we used the same architecture for each model type (CNN, ResNet, RNN) across both static and DBSA experiments. The only difference between the Static and DBSA versions is the batch size strategy: static runs use a fixed batch size of 512, while DBSA starts at 512 and adjusts it dynamically based on loss and gradient behavior. Everything else—network layers, parameters, learning rates, and optimization algorithms—remains identical within each model type (e.g., CNN Static vs. CNN DBSA). This controlled setup ensures that any performance differences (e.g., accuracy, loss, convergence speed) come purely from the batch size approach, not architectural variations. By doing this, we can clearly measure DBSA’s advantage in adapting batch size, making our results a direct test of its effectiveness across diverse models and datasets.

**Prerequisites**
To run the code, you’ll need:

**Python:** Version 3.8 or higher
**Libraries:**
PyTorch (pip install torch torchvision)
NumPy (pip install numpy)
Matplotlib (for visualization, pip install matplotlib)
**Datasets:** CIFAR-10 and MNIST are downloaded via PyTorch’s torchvision.datasets.
**Environment:** Google Colab (recommended) or a local Python setup with GPU support.


