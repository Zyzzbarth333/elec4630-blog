---
layout: post
title: "GPU Acceleration: Unravelling Deep Learning Performance Gains"
date: 2025-04-16 18:00:00 +1000
categories: [deep-learning, hardware, performance]
tags: [gpu, cpu, container, docker, performance-analysis]
toc: true
use_math: true
image: ../images/Q3/GPU-batch-test.jpg
description: "A deep dive into GPU acceleration for deep learning, exploring batch size optimisation and performance characteristics using NVIDIA RTX 3060 Ti."
---

# Accelerating Deep Learning: A Practical GPU Performance Analysis

## Introduction

Deep learning models, particularly Convolutional Neural Networks (CNNs), rely heavily on matrix computations that can benefit dramatically from parallel processing. In this post, I'll share my exploration of GPU acceleration during the ELEC4630 Computer Vision and Deep Learning course, focusing on understanding how hardware can dramatically improve training efficiency.

## Experimental Setup

### Hardware Configuration

- **CPU**: Intel Core i7-11700K (16 cores, 3.60 GHz)
- **GPU**: NVIDIA GeForce RTX 3060 Ti (8GB VRAM)
- **System**: WSL2 with Ubuntu 22.04

### Methodology

I designed a systematic experiment to test how batch size impacts GPU training performance:

1. Use a consistent dataset of bird and woodland images
2. Apply transfer learning with ResNet-18
3. Test batch sizes: 16, 32, 64, 128, and 256
4. Measure precise training times
5. Monitor GPU utilisation using `nvtop`

## Batch Size Performance Results

| Batch Size | Training Time | Relative Performance |
|-----------|--------------|----------------------|
| 16        | 9.44s        | 76.0%                |
| 32        | 10.50s       | 68.4%                |
| 64        | 7.18s        | 100.0% (Optimal)     |
| 128       | 10.52s       | 68.3%                |
| 256       | 11.35s       | 63.3%                |

### Performance Visualisation

![Batch Size Performance](/images/Q3/batch_size_performance.png)

## Key Insights

### 1. U-Shaped Performance Curve

The results reveal a classic U-shaped performance curve:

- **Small Batch Sizes (16, 32)**: 
  - Underutilise GPU parallel processing
  - Increased data loading overhead
  - More synchronisation points per epoch

- **Optimal Batch Size (64)**:
  - Best balance between parallelisation and overhead
  - Keeps GPU execution units efficiently busy
  - Minimises memory pressure

- **Large Batch Sizes (128, 256)**:
  - Create memory access bottlenecks
  - Reduce computational efficiency
  - Higher initial loss values

### 2. GPU Utilisation Patterns

Monitoring with `nvtop` revealed fascinating insights:

![GPU Monitoring](/images/Q3/GPU-batch-test.jpg)

- Blue line: GPU computational utilisation (0-100%)
- Yellow line: Memory usage

Key observations:
- Optimal batch size shows sustained high utilisation
- Smaller batches create more idle periods
- Larger batches approach memory bandwidth limits

## Performance Speedup

The most critical metric: **GPU vs CPU Comparison**

- **CPU Training Time**: 71.60 seconds
- **GPU Training Time**: 7.18 seconds
- **Speedup Factor**: 9.97×

This means a task taking 1.2 minutes on CPU completes in just 0.1 minutes on GPU!

## Mathematical Model

I'm developing a mathematical model to describe batch size performance:

$$T(b) = \frac{c}{b^\alpha} + T_{min}$$

Where:
- $T(b)$ is training time for batch size $b$
- $c$ is a constant factor
- $\alpha$ is the scaling factor (typically $0 < \alpha < 1$)
- $T_{min}$ is the minimum possible training time

## Practical Recommendations

1. **Batch Size Optimization**
   - Start with a batch size of 64 for similar hardware
   - Experiment within the 32-128 range
   - Monitor GPU metrics during training

2. **Hardware Considerations**
   - Invest in GPU acceleration
   - Consider memory capacity
   - Balance computational efficiency with memory constraints

## Containerization Approach

I used Docker containers with NVIDIA Container Toolkit to ensure:
- Consistent environment
- Easy switching between CPU and GPU configurations
- Reproducible experimental setup

## Future Work

Planned extensions to this analysis:
- Investigate mixed-precision training
- Test with larger, more complex models
- Explore advanced data pipeline optimisations

## Conclusion

GPU acceleration offers transformative performance gains for deep learning, with batch size crucial in optimisation. Our 9.97× speedup demonstrates the practical importance of understanding hardware-software interactions.

## References

1. Lovell, B. (2025). "BYOD Image for Deep Learning"
2. Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with fastai and PyTorch*
3. Kandel, I., & Castelli, M. (2020). "The effect of batch size on the generalizability of convolutional neural networks"

*Curious about GPU acceleration? Let's discuss in the comments!*
