---
layout: post
title: "Multi-Class Image Classification with t-SNE Visualisation"
date: 2025-04-18 16:30:00 +1000
categories: [deep-learning, computer-vision]
tags: [fastai, t-sne, classification, web-scraping, confusion-matrix]
toc: true
use_math: true
---

# Multi-Class Image Classification with t-SNE Visualisation

For Question 4 of ELEC4630 Assignment 2, I'm developing a multi-class image classifier to distinguish between five categories: airplanes, automobiles, birds, cats, and dogs. This post documents my approach to web scraping for dataset creation, model training, and performance analysis using t-SNE visualisation and confusion matrices.

## Introduction

Image classification has become a cornerstone application of deep learning, with impressive results across various domains. This project explores building a custom classifier using fastai and PyTorch, with a particular focus on:

1. Creating a dataset by scraping images from the web
2. Training an effective multi-class classifier
3. Visualising high-dimensional features using t-SNE
4. Analysing model performance with confusion matrices

This approach demonstrates how we can quickly build and evaluate custom image classifiers without relying on pre-existing datasets like CIFAR-10 or ImageNet.

## Dataset Creation via Web Scraping

Following the methodology from the fastai course example "00-is-it-a-bird-creating-a-model-from-your-own-data.ipynb", I'm using DuckDuckGo to scrape images for each category.

### Scraping Implementation (Work in Progress)

```python
# TODO: Finalise this code
from fastai.vision.all import *
from duckduckgo_search import ddg_images
from fastdownload import download_url
import time
from fastcore.all import *

# Function to search for images and download them
def search_images(term, max_images=200):
    print(f"Searching for '{term}'")
    # Using DuckDuckGo search API to find images
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

# Function to download and save images
def download_images(urls, dest_dir, max_pics=200, timeout=4):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    
    downloaded = 0
    for i, url in enumerate(urls):
        if downloaded >= max_pics: break
        try:
            # Download and save the image
            fname = dest_dir/f'{i:04d}.jpg'
            if not fname.exists():
                download_url(url, fname, timeout=timeout)
                downloaded += 1
                if i%20 == 0: print(f'{i} images downloaded to {dest_dir.name}')
        except Exception as e:
            print(f'Error downloading {url}: {e}')
    
    return downloaded

# Main function to create the dataset
def create_dataset():
    # Define the classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'dog']
    
    # Create directories
    path = Path('animal_classification')
    for c in classes:
        (path/c).mkdir(exist_ok=True, parents=True)
    
    # Search terms with variations to improve diversity
    search_terms = {
        'airplane': ['commercial airplane', 'passenger aircraft', 'military airplane', 'private jet'],
        'automobile': ['car photo', 'sedan', 'SUV', 'sports car'],
        'bird': ['wild bird', 'colorful bird', 'flying bird', 'bird perching'],
        'cat': ['cat pet', 'domestic cat', 'cat portrait', 'kitten'],
        'dog': ['dog pet', 'dog portrait', 'puppy', 'large dog']
    }
    
    # TODO: Download images for each class using multiple search terms
    # for cls in classes:
    #     for term in search_terms[cls]:
    #         urls = search_images(term, max_images=50)
    #         download_images(urls, path/cls, max_pics=50)
    
    # TODO: Implement image cleaning/verification
    # - Remove corrupted images
    # - Verify minimum dimensions
    # - Ensure consistent formats
    
    return path

# TODO: Call the function to create the dataset
# path = create_dataset()
```

This code structure allows for flexible dataset creation, with search term variations to improve dataset diversity. The current implementation needs to be refined to handle potential errors and ensure image quality.

## Data Preprocessing and Augmentation

Before training, the images require preprocessing and augmentation to improve model generalisation:

```python
# TODO: Implement proper data preprocessing and augmentation
def prepare_data(path, img_size=224, batch_size=64, valid_pct=0.2):
    # Define transformations
    item_tfms = [
        Resize(img_size),           # Resize to consistent dimensions
        # TODO: Add more item transforms if needed
    ]
    
    batch_tfms = [
        *aug_transforms(size=img_size, min_scale=0.8),  # Standard augmentations
        Normalize.from_stats(*imagenet_stats),         # Normalize using ImageNet stats
    ]
    
    # Create DataLoaders
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=valid_pct,
        seed=42,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=batch_size
    )
    
    return dls

# TODO: Call the function to prepare data
# dls = prepare_data(path)
# dls.show_batch(max_n=16, figsize=(10,10))
```

The preprocessing pipeline resizes images to a consistent dimension and applies standard augmentations like random cropping, flipping, and rotation. These transformations help the model generalise better to unseen images.

## Model Architecture and Training

For this classification task, I'm using a ResNet-50 model pretrained on ImageNet, fine-tuned for our specific classes:

```python
# TODO: Implement model training with appropriate loss function
def train_model(dls, epochs=10, learning_rate=1e-3):
    # Create the learner with ResNet-50 architecture
    learn = vision_learner(
        dls, 
        resnet50,  # Using ResNet-50 as base architecture
        metrics=[accuracy, error_rate],
        loss_func=CrossEntropyLossFlat()  # Standard multiclass classification loss
    )
    
    # Find optimal learning rate
    # TODO: Implement learning rate finder
    # learn.lr_find()
    
    # Train the model using 1cycle policy
    # TODO: Implement proper training
    # learn.fine_tune(epochs, learning_rate)
    
    return learn

# TODO: Call the function to train the model
# learn = train_model(dls)
```

### Loss Function Discussion

For this multi-class classification problem, I'm using the standard Cross-Entropy Loss function, which is well-suited for exclusive classification problems where each image belongs to exactly one class. The mathematical form of this loss is:

$$\mathcal{L}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

Where:
- $C$ is the number of classes (5 in our case)
- $y_i$ is the ground truth (1 for the correct class, 0 otherwise)
- $\hat{y}_i$ is the predicted probability for class $i$

The fastai implementation `CrossEntropyLossFlat()` handles the proper formatting of inputs and targets, making it easy to use with the training pipeline.

## Performance Evaluation (To Be Implemented)

After training, comprehensive evaluation is needed to understand model performance:

```python
# TODO: Implement model evaluation and metrics calculation
def evaluate_model(learn):
    # Get predictions on validation set
    preds, targets = learn.get_preds()
    
    # Calculate overall accuracy
    accuracy = (preds.argmax(dim=1) == targets).float().mean()
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Calculate per-class metrics
    # TODO: Implement precision, recall, and F1 score for each class
    
    # Create confusion matrix
    # TODO: Implement confusion matrix visualisation
    
    return preds, targets

# TODO: Call the function to evaluate the model
# preds, targets = evaluate_model(learn)
```

## t-SNE Visualisation

One of the most powerful ways to understand high-dimensional data is through t-SNE (t-distributed Stochastic Neighbour Embedding), which projects complex feature representations into a 2D space while preserving local relationships:

```python
# TODO: Implement t-SNE visualisation
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_tsne(learn, perplexity=30, n_iter=1000):
    # Extract features from the penultimate layer
    # TODO: Implement feature extraction
    # activation_extractor = ActivationExtractor(learn.model, learn.model[1][-1])
    # features, targets = [], []
    # with ActivationHook(learn.model, activation_extractor) as hook:
    #     preds, actual_targets = learn.get_preds(with_decoded=True)
    #     features = hook.stored_activations
    #     targets = actual_targets
    
    # Convert to numpy arrays
    # features_np = features.cpu().numpy()
    # targets_np = targets.cpu().numpy()
    
    # Apply t-SNE dimensionality reduction
    # tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    # tsne_result = tsne.fit_transform(features_np)
    
    # Create visualization
    # plt.figure(figsize=(12, 10))
    # classes = learn.dls.vocab
    # colors = sns.color_palette("bright", len(classes))
    
    # for i, cls in enumerate(classes):
    #     indices = np.where(targets_np == i)[0]
    #     plt.scatter(
    #         tsne_result[indices, 0],
    #         tsne_result[indices, 1],
    #         c=[colors[i]],
    #         label=cls,
    #         alpha=0.7
    #     )
    
    # plt.legend()
    # plt.title('t-SNE Visualization of Image Features')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return None  # Will return the t-SNE result when implemented

# TODO: Call the function to create t-SNE visualization
# tsne_result = visualize_tsne(learn)
```

This visualisation will help us understand:
1. How well the classes are separated in feature space
2. Which classes are most similar or confusable
3. Whether there are subgroups within classes
4. Potential outliers or mislabeled examples

## Confusion Matrix Analysis

The confusion matrix provides a detailed view of correct and incorrect classifications across all classes:

```python
# TODO: Implement confusion matrix visualisation and analysis
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(preds, targets, class_names):
    # Get class predictions from probabilities
    pred_classes = preds.argmax(dim=1).cpu().numpy()
    true_classes = targets.cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    
    # Normalise by row (true classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create visualisation
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis of most confused classes
    # TODO: Implement analysis of confused pairs
    
    return cm, cm_norm

# TODO: Call the function to visualize confusion matrix
# cm, cm_norm = plot_confusion_matrix(preds, targets, learn.dls.vocab)
```

The confusion matrix will reveal:
1. Which classes are most often confused with each other
2. The balance of true positives, false positives, and false negatives
3. Whether errors are distributed evenly or concentrated on specific classes

## Current Progress and Challenges

This project is still in development, with several key components to be implemented and refined:

1. **Dataset Creation:** Need to complete image scraping and cleaning
2. **Model Training:** Need to tune hyperparameters and evaluate different architectures
3. **Feature Visualisation:** Need to implement t-SNE properly with feature extraction
4. **Performance Analysis:** Need to compute comprehensive metrics and analysis

Some challenges I'm currently addressing:

1. **Data Quality:** Web-scraped images often contain irrelevant content or incorrect labels
2. **Class Imbalance:** Ensuring balanced representation across all five classes
3. **t-SNE Implementation:** Extracting the right features from the model for visualisation
4. **Performance Optimisation:** Finding the right balance of accuracy and computational efficiency

## Next Steps

To complete this project, I plan to:

1. Finish dataset creation with thorough cleaning and validation
2. Implement proper model training with hyperparameter tuning
3. Complete t-SNE visualisation implementation
4. Generate comprehensive performance metrics and analysis
5. Investigate opportunities for model improvement

## Conclusion (Preliminary)

Building a custom image classifier from web-scraped data presents unique challenges but offers valuable insights into the complete machine learning pipeline. The combination of modern deep learning techniques (transfer learning, data augmentation) with visualisation tools (t-SNE, confusion matrices) provides a powerful approach to understanding model behaviour.

The preliminary framework established in this post demonstrates how to approach multi-class image classification problems systematically, from data collection through to performance evaluation. As this project progresses, I'll update with full implementation details and results.

## References

1. Howard, J., & Gugger, S. (2020). "Deep Learning for Coders with fastai and PyTorch." O'Reilly Media.
2. Van der Maaten, L., & Hinton, G. (2008). "Visualising data using t-SNE." Journal of Machine Learning Research, 9(Nov), 2579-2605.
3. fastai Documentation: [https://docs.fast.ai/](https://docs.fast.ai/)
4. DuckDuckGo Search API: [https://github.com/deepanprabhu/duckduckgo-images-api](https://github.com/deepanprabhu/duckduckgo-images-api)
5. Wattenberg, M., Viégas, F., & Johnson, I. (2016). "How to Use t-SNE Effectively." Distill. [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
