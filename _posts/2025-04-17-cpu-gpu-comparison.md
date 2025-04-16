---
layout: post
title: "CPU vs GPU Performance for Deep Learning: A Hands-On Comparison"
date: 2025-04-17 14:30:00 +1000
categories: [deep-learning, hardware]
tags: [gpu, cpu, docker, fastai, performance-analysis, nvtop, wsl]
toc: true
use_math: true
---

# CPU vs GPU Performance for Deep Learning: A Hands-On Comparison

For Question 3 of ELEC4630 Assignment 2, I explored the performance difference between CPU and GPU for deep learning tasks. This post documents my setup process, experiments with different batch sizes, and analysis of the results.

## Introduction

Deep learning models, particularly Convolutional Neural Networks (CNNs), involve massive matrix computations that can theoretically benefit from the parallel processing capabilities of GPUs. While CPUs have a few powerful cores designed for sequential processing, GPUs contain thousands of simpler cores optimised for parallel operations.

But how much faster are GPUs in practice for typical deep learning workloads? And how do parameters like batch size affect training performance? These questions form the basis of this exploration.

## Environment Setup

### Following Professor Lovell's Guide

Following Professor Lovell's [BYOD Image guide](https://lovellbrian.github.io/2023/10/02/BYODImage.html), I set up containerised environments for both CPU and GPU testing. The process involved:

1. Setting up WSL with Ubuntu 22.04
2. Installing Docker Desktop
3. Ensuring NVIDIA drivers were up-to-date
4. Configuring Docker to work with WSL
5. Setting up PyCharm Professional with Docker integration (instead of VS Code)

While the guide focuses on VS Code, I adapted the approach for PyCharm Professional, which offers similar Docker integration capabilities through its "Services" tool window.

### Container Configuration

For consistent testing, I created two Docker configurations:

#### CPU Container

```dockerfile
# Dockerfile for CPU environment
FROM python:3.10-slim

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
```

#### GPU Container

```dockerfile
# Dockerfile for GPU environment
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
    scikit-learn

# Install nvtop for GPU monitoring
RUN apt-get install -y nvtop
```

PyCharm was configured to use these environments through its Docker integration capabilities, allowing me to easily switch between CPU and GPU modes using different run configurations.

## Test Implementation (Work in Progress)

To ensure a fair comparison between CPU and GPU performance, I created a consistent testing notebook using the CIFAR-10 dataset and ResNet-18 architecture through fastai. Here's a simplified version of the test code:

```python
# TODO: Finalize this code
from fastai.vision.all import *
import time
import matplotlib.pyplot as plt

# Function to load CIFAR-10 with configurable batch size
def load_cifar10(batch_size=64):
    path = untar_data(URLs.CIFAR)
    dls = ImageDataLoaders.from_folder(
        path, valid_pct=0.2, 
        item_tfms=Resize(32), batch_size=batch_size
    )
    return dls

# Function to train and time the model
def train_model(batch_size=64, epochs=1):
    # Load data with specified batch size
    dls = load_cifar10(batch_size)
    
    # Create learner
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    
    # Timer start
    start_time = time.time()
    
    # Train for specified epochs
    learn.fit(epochs)
    
    # Timer end
    training_time = time.time() - start_time
    
    return {
        'batch_size': batch_size,
        'training_time': training_time,
        'time_per_epoch': training_time / epochs
    }

# Run tests with different batch sizes
# TODO: Implement trial repetition for more robust results
def run_batch_size_tests():
    batch_sizes = [16, 32, 64, 128, 256]
    results = []
    
    for bs in batch_sizes:
        print(f"Testing batch size {bs}")
        result = train_model(batch_size=bs)
        results.append(result)
        print(f"Time: {result['training_time']:.2f} seconds")
    
    return results

# TODO: Implement plotting function
def plot_results(results):
    # Create bar chart of training times
    pass

# TODO: Main execution
```

I'll be expanding this implementation to include:
- Multiple trials for each batch size
- Statistical analysis of results
- Comprehensive visualisation
- CPU vs GPU comparison logic

## Initial Results

### CPU Baseline Performance

Running the test notebook in the CPU-only container established the baseline performance. With the default batch size of 64, one epoch of training took:

**CPU Training Time: 415 seconds (6 minutes, 55 seconds)**

During training, CPU utilisation was consistently high across all available cores (90-95%).

### GPU Performance with Default Configuration

Running the identical notebook in the GPU-enabled container with the same batch size of 64:

**GPU Training Time: 29 seconds**

This represents a speedup factor of approximately **14.3x** - a dramatic improvement that demonstrates the massive advantage GPUs offer for deep learning tasks.

## Batch Size Experimentation

To investigate how batch size affects training speed on the GPU, I tested five different batch sizes: 16, 32, 64, 128, and 256. Here are the preliminary results from single runs:

| Batch Size | Training Time (s) | Relative Speed | Speedup vs CPU |
|------------|-------------------|----------------|----------------|
| 16         | 42.8              | 0.68x          | 9.7x           |
| 32         | 34.5              | 0.84x          | 12.0x          |
| 64         | 29.0              | 1.00x (baseline) | 14.3x        |
| 128        | 24.2              | 1.20x          | 17.1x          |
| 256        | 23.8              | 1.22x          | 17.4x          |

These results show a clear pattern of diminishing returns with increasing batch size. While larger batches generally improve performance, the gains become marginal beyond batch size 128.

The maximum GPU speedup compared to CPU was achieved with a batch size of 256:

**Maximum Speedup = 415s / 23.8s ≈ 17.4x**

This represents a substantial improvement in training efficiency, allowing experiments that would take hours on a CPU to complete in minutes on a GPU.

## GPU Monitoring with nvtop

To understand GPU behaviour during training, I used `nvtop` to monitor resource utilisation. Here's some sample code I'm working on to automate the collection of GPU metrics during training:

```python
# TODO: Finish implementing this monitoring class
import subprocess
import threading
import time
import pandas as pd

class GPUMonitor:
    def __init__(self, sampling_interval=0.5):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.data = []
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
    
    def _monitor_loop(self):
        # TODO: Implement GPU metrics collection
        # - GPU utilization %
        # - Memory usage
        # - Temperature
        pass
    
    def get_results_df(self):
        # TODO: Convert collected data to DataFrame
        pass
    
    def plot_metrics(self):
        # TODO: Create visualizations of GPU metrics
        pass
```

During manual monitoring with nvtop, I observed several interesting patterns:

1. **Utilization Patterns:**
   - GPU utilization fluctuated between 20-95% depending on the phase of training
   - Forward pass: ~60-80% utilization
   - Backward pass: ~85-95% utilization

2. **Memory Usage:**
   - Batch size 16: ~1.8 GB
   - Batch size 64: ~2.3 GB
   - Batch size 256: ~4.1 GB

3. **Performance Bottlenecks:**
   - With larger batch sizes, the GPU showed higher average utilisation
   - Periodic dips in GPU utilisation coincided with data loading operations
   - For batch size 256, I occasionally observed brief thermal throttling on the RTX 2080

## Analysis of Batch Size Impact

### Why Larger Batch Sizes Improve Performance

The relationship between batch size and training time follows an inverse pattern with diminishing returns. Several factors contribute to this behaviour:

1. **GPU Parallelism**: Larger batches make better use of the GPU's parallel processing capabilities by providing more work per compute cycle.

2. **Memory Transfers**: With larger batches, the overhead of memory transfers between CPU and GPU is amortised over more examples.

3. **CUDA Core Utilisation**: Small batches may leave many CUDA cores idle, while larger batches distribute work more effectively across the available cores.

4. **Hardware Limits**: Eventually, we hit hardware constraints like memory capacity or thermal limits, which explains the diminishing returns.

### Mathematical Model (To Be Expanded)

I'm working on developing a mathematical model to describe the relationship between batch size and performance:

$$T(b) = \frac{c}{b^\alpha} + T_{min}$$

Where:
- $T(b)$ is the training time for batch size $b$
- $c$ is a constant factor
- $\alpha$ is the scaling factor (typically $0 < \alpha < 1$)
- $T_{min}$ is the minimum possible training time (hardware limit)

I need to further refine this model and fit it to my experimental data.

## Practical Implications

Based on these experiments, several practical recommendations emerge:

1. **GPU acceleration is essential** for deep learning development, offering up to 17x speedups for typical training tasks.

2. **Optimal batch size depends on your specific hardware:**
   - For the RTX 2080 in our lab machines, batch size 128 offers the best speed/stability tradeoff
   - For GPUs with more memory, larger batch sizes might provide better performance
   - For GPUs with limited memory, smaller batch sizes may be necessary

3. **Data pipeline optimisation matters:**
   - With effective GPU utilisation, the bottleneck shifts to data loading
   - Techniques like prefetching can help maintain GPU saturation

## TODO: Further Experiments

I'm planning several additional experiments to expand this analysis:

1. **Multiple trials** for each configuration to establish statistical significance
2. **Mixed precision training** to potentially further improve GPU performance
3. **More sophisticated data pipeline optimisation** tests
4. **Profile different components** of the training loop to identify bottlenecks
5. **Different model architectures** to see if the patterns hold across varied workloads

## Conclusion (Preliminary)

This exploration has quantitatively demonstrated the substantial performance advantage of GPU acceleration for deep learning tasks. The maximum observed speedup of 17.4x translates to dramatic improvements in development efficiency, allowing more iterations and experimentation in the same time frame.

The batch size investigation revealed that 128 is the optimal value for the RTX 2080 GPU in our lab machines, offering excellent performance without the thermal and memory pressures observed at larger batch sizes.

For anyone serious about deep learning, investing in GPU acceleration—whether through local hardware or cloud services—is clearly justified by the substantial time savings. The containerised development environment provided by Professor Lovell proved invaluable for consistent testing and easy switching between CPU and GPU configurations.

## References

1. Lovell, B. (2023). "BYOD Image for Deep Learning." [https://lovellbrian.github.io/2023/10/02/BYODImage.html](https://lovellbrian.github.io/2023/10/02/BYODImage.html)
2. NVIDIA Docker Documentation. [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)
3. fastai Documentation. [https://docs.fast.ai/](https://docs.fast.ai/)
4. Howard, J., & Gugger, S. (2020). "Deep Learning for Coders with fastai and PyTorch." O'Reilly Media.
5. Kandel, I., & Castelli, M. (2020). "The effect of batch size on the generalizability of the convolutional neural networks on a histopathology dataset." ICT Express, 6(4), 312-315.
