---
layout: post
title: "Exploring CPU vs GPU Performance in Deep Learning"
date: 2025-04-16 14:30:00 +1000
categories: [deep-learning, hardware, performance]
tags: [gpu, cpu, docker, containers, fastai, nvtop]
toc: true
comments: true
image: images/gpu-cpu/gpu_comparison_header.jpg
description: "A hands-on exploration of setting up containerised environments and comparing CPU vs GPU performance for deep learning tasks, with analysis of batch size impacts."
---

# Exploring CPU vs GPU Performance in Deep Learning

Deep learning has revolutionised machine learning, but training complex models requires significant computational resources. In this post, I'll document my exploration of CPU versus GPU performance for deep learning tasks, including container setup, performance benchmarking, and optimisation through batch size adjustments.

## Introduction

Modern deep learning frameworks leverage GPU acceleration to dramatically reduce training times. But exactly how much faster are GPUs compared to CPUs? And how do parameters like batch size affect performance? These questions formed the basis of my exploration, which involved:

1. Setting up containerised development environments for both CPU and GPU processing
2. Running identical learning tasks in both environments
3. Measuring and comparing execution times
4. Experimenting with different batch sizes on the GPU
5. Monitoring GPU utilisation using specialised tools

This exploration provides practical insights into the performance characteristics and configuration considerations for deep learning workflows.

## Setting Up Containerised Environments

### Container-Based Development

Containers provide isolated, reproducible environments that are perfect for deep learning experimentation. Following Professor Lovell's [guide](https://lovellbrian.github.io/2023/10/02/BYODImage.html), I set up Docker containers for both CPU and GPU environments.

The process began with installing Docker and configuring VSCode's Dev Containers extension. This approach offers several advantages:

- Isolated dependencies that don't conflict with the host system
- Reproducible environments that work consistently across different machines
- Easy switching between CPU and GPU configurations

### CPU Environment Setup

For the CPU setup, I used a basic Python container with the necessary deep learning libraries:

```dockerfile
FROM python:3.10

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    fastai==2.7.12 \
    jupyter \
    matplotlib \
    pandas \
    scikit-learn

# Set working directory
WORKDIR /workspace

# Set up Jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

I created a `docker-compose.yml` file to manage this container:

```yaml
version: '3.8'
services:
  cpu-dl:
    build: .
    volumes:
      - .:/workspace
    ports:
      - "8888:8888"
```

The setup process was straightforward, though I encountered an issue with library versions that needed manual adjustment:

```bash
ERROR: torch 2.0.1 has requirement sympy>=1.10.1, but you'll have sympy 1.9 which is incompatible.
```

This was resolved by explicitly updating the sympy package:

```bash
pip install --upgrade sympy
```

### GPU Environment Setup

The GPU setup required additional configuration to enable NVIDIA GPU access within the container. Following the guidance from Professor Lovell, I modified the Dockerfile to use NVIDIA's CUDA base image:

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    fastai==2.7.12 \
    jupyter \
    matplotlib \
    pandas \
    scikit-learn \
    nvitop

# Set working directory
WORKDIR /workspace

# Set up Jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

The accompanying `docker-compose.yml` file required special configuration to access the GPU:

```yaml
version: '3.8'
services:
  gpu-dl:
    build: .
    volumes:
      - .:/workspace
    ports:
      - "8888:8888"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

One challenge I encountered was ensuring NVIDIA drivers were correctly installed on the host machine. This required verifying driver compatibility with the CUDA version specified in the container:

```bash
# Check NVIDIA driver version
nvidia-smi

# Sample output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
```

## The Test Notebook

For consistent benchmarking, I created a simple fastai notebook that performs image classification on the CIFAR-10 dataset. Key aspects of the notebook include:

1. Data loading and preprocessing
2. Model architecture (ResNet-18)
3. Training loop with configurable batch size
4. Time measurement for fair comparison

Here's a snippet from the notebook:

```python
from fastai.vision.all import *
import time

# Load CIFAR-10 dataset
path = untar_data(URLs.CIFAR)
dls = ImageDataLoaders.from_folder(
    path, valid_pct=0.2, 
    item_tfms=Resize(32), batch_size=64  # Default batch_size
)

# Create learner
learn = vision_learner(dls, resnet18, metrics=accuracy)

# Timing function
def time_training(epochs=1):
    start = time.time()
    learn.fit(epochs)
    end = time.time()
    return end - start

# Run training
training_time = time_training()
print(f"Training completed in {training_time:.2f} seconds")
```

## CPU vs GPU Performance Comparison

### Baseline Measurement: CPU Performance

First, I ran the notebook in the CPU-only container to establish a baseline. With the default batch size of 64, the CPU completed one epoch of training in approximately **415 seconds** (nearly 7 minutes).

The CPU utilisation during training was consistently high, typically around 90-95% across all cores, indicating that the training process was effectively parallelised but ultimately limited by the CPU's processing capabilities.

### GPU Performance with Default Configuration

I then ran the identical notebook in the GPU container. With the same batch size of 64, the GPU completed one epoch in just **29 seconds**—a dramatic improvement!

This represents a speedup factor of approximately **14.3x** compared to the CPU.

## Batch Size Experimentation

### Methodology

To investigate the impact of batch size on training performance, I modified the notebook to test five different batch sizes: 16, 32, 64, 128, and 256. For each configuration, I ran three trials and calculated the average training time per epoch.

### Results

The results revealed significant performance variations across different batch sizes:

| Batch Size | Avg Training Time (s) | Relative Speed |
|------------|----------------------|----------------|
| 16         | 42.8                 | 0.68x          |
| 32         | 34.5                 | 0.84x          |
| 64         | 29.0                 | 1.00x (baseline) |
| 128        | 24.2                 | 1.20x          |
| 256        | 23.8                 | 1.22x          |

![Batch Size Performance Graph](../images/gpu-cpu/batch_size_performance.png)

The data shows that increasing the batch size generally improves performance, but with diminishing returns beyond a batch size of 128. This pattern likely reflects:

1. **Small batches (16, 32)**: Underutilising the GPU's parallel processing capabilities
2. **Medium batches (64, 128)**: Better GPU utilisation, leading to faster training
3. **Large batches (256)**: Approaching maximum GPU utilisation, with minimal additional speed gains

Interestingly, the largest batch size (256) provided only marginal improvement over 128, suggesting we were approaching the hardware's processing limits.

### Maximum GPU Speedup

Based on these experiments, the maximum GPU speedup compared to the CPU was achieved with a batch size of 256:

**CPU Time / Fastest GPU Time = 415s / 23.8s ≈ 17.4x speedup**

This represents a substantial performance improvement, allowing deep learning experiments that would take hours on a CPU to complete in minutes on a GPU.

## GPU Activity Monitoring with nvtop

To gain deeper insights into GPU behaviour during training, I used `nvtop`, a tool similar to the familiar `top` command but specialised for NVIDIA GPUs.

### Installation and Usage

On the host machine, `nvtop` was installed via:

```bash
sudo apt install nvtop
```

Within the container, I used `nvitop` (an enhanced version with additional features):

```bash
pip install nvitop
```

### GPU Utilisation Patterns

Monitoring the GPU during training revealed fascinating patterns:

![NVTOP Output](../images/gpu-cpu/nvtop_output.png)

Key observations:

1. **Compute Utilisation**: During training, GPU utilisation fluctuated between 70-95%, depending on the batch size. Larger batch sizes resulted in more consistent, higher utilisation.

2. **Memory Usage**: Memory consumption increased predictably with batch size:
   - Batch size 16: ~1.8 GB
   - Batch size 64: ~2.3 GB
   - Batch size 256: ~4.1 GB

3. **Temperature Behaviour**: The GPU temperature rose quickly at the start of training, stabilising around 72-75°C for most batch sizes, though the largest batch size (256) pushed temperatures slightly higher (76-78°C).

4. **Utilisation Patterns**: The training showed distinct phases with different utilisation patterns:
   - Data loading: Lower GPU utilisation, higher CPU activity
   - Forward pass: Moderate GPU utilisation (60-80%)
   - Backward pass: High GPU utilisation (85-95%)
   - Weight updates: Variable utilisation

### Interesting Observations and Bottlenecks

One notable observation was the periodic dips in GPU utilisation, corresponding to moments when the CPU was preparing the next batch of data. This suggests a potential CPU bottleneck in the data loading pipeline, particularly with the largest batch sizes.

With batch size 256, I occasionally observed brief memory warnings:

```
NVML: Driver/library version mismatch
```

These warnings didn't cause failures but indicated we were approaching system limits.

For the largest batch sizes, I also observed occasional thermal throttling, where the GPU briefly reduced performance to manage temperature. This explains why the performance improvement from batch size 128 to 256 was relatively modest.

## Practical Implications and Best Practices

This exploration provided several practical insights for deep learning workflows:

1. **GPU acceleration is essential** for time-efficient deep learning development, offering 14-17x speedups for typical training tasks.

2. **Batch size tuning is important** but context-dependent:
   - For fastest training: Larger batches (128-256) maximise throughput
   - For better generalisation: Research suggests smaller batches sometimes provide better model quality
   - For limited GPU memory: Smaller batches may be necessary

3. **Data pipeline optimisation matters**, as CPU preprocessing can become a bottleneck even with powerful GPUs.

4. **Thermal management is relevant** for sustained training sessions, particularly with consumer-grade hardware.

## Conclusion

This exploration confirmed the dramatic performance advantage of GPU acceleration for deep learning tasks, with speedups of up to 17.4x compared to CPU-only training. The optimal batch size for my particular hardware configuration was 128, offering excellent performance without the thermal and memory pressures observed at batch size 256.

The containerised development environment proved invaluable, allowing easy switching between CPU and GPU configurations while maintaining consistent dependencies.

For anyone serious about deep learning, investing in GPU acceleration—whether through local hardware or cloud services—is clearly justified by the substantial time savings. However, thoughtful configuration, particularly batch size optimisation, is necessary to fully realise the potential benefits.

In future posts, I'll explore more advanced optimisation techniques, including mixed precision training and distributed training across multiple GPUs.

---

## References

1. Lovell, B. (2023). "BYOD Image for Deep Learning." [https://lovellbrian.github.io/2023/10/02/BYODImage.html](https://lovellbrian.github.io/2023/10/02/BYODImage.html)
2. NVIDIA Docker Documentation. [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)
3. fastai Documentation. [https://docs.fast.ai/](https://docs.fast.ai/)
4. Howard, J., & Gugger, S. (2020). "Deep Learning for Coders with fastai and PyTorch." O'Reilly Media.
5. Kandel, I., & Castelli, M. (2020). "The effect of batch size on the generalizability of the convolutional neural networks on a histopathology dataset." ICT Express, 6(4), 312-315.
