---
layout: post
title: "Multi-Class Image Classification with t-SNE Visualisation"
date: 2025-04-20 18:00:00 +1000
categories: [deep-learning, computer-vision]
tags: [fastai, t-sne, classification, web-scraping, confusion-matrix]
toc: true
use_math: true
image: /elec4630-blog/images/sample_batch.png
description: "Building a multi-class image classifier from web-scraped data with comprehensive performance analysis using t-SNE visualisation and confusion matrices."
---

# Building a Multi-Class Image Classifier: From Web Scraping to Deep Learning

Have you ever wondered how to build an image classifier from scratch using only web-scraped data? In this post, I'll share my journey creating a five-class image classification system as part of my ELEC4630 computer vision coursework. Rather than using a standard dataset, I built everything from the ground up using images found online!

This project challenged me to classify images into five everyday categories: airplanes, automobiles, birds, cats, and dogs. Let's dive into the process and see what I discovered.

## The Data Challenge: Building a Dataset from Scratch

One of the most exciting (and challenging!) parts of this project was creating my own dataset using web-scraped images. Most tutorials rely on carefully curated datasets like CIFAR-10 or ImageNet, but I wanted to see if I could build a robust classifier using only images found "in the wild."

> 📝 **Note:** The full implementation details are available in my [Jupyter notebook (Q4-notebook.pdf)](https://github.com/zyzzbarth333/elec4630-blog/assets/Q4-notebook.pdf) if you're interested in exploring the code in more depth!

### Web Scraping Strategy

I used DuckDuckGo's search API to collect images for each category, employing a clever strategy to ensure diversity:

```python
# Define the five classes for our multi-class classifier
classes = ["airplane", "automobile", "bird", "cat", "dog"]

# For each class, use multiple search terms to ensure diversity
search_terms = [
    f"{cls} photo",
    f"{cls} sun photo",
    f"{cls} shade photo"
]
```

Why multiple search terms? I discovered that variations like "sun photo" and "shade photo" brought in images with different lighting conditions and contexts, creating a more robust dataset.

After cleaning up problematic files and removing any corrupted images, I ended up with a nicely balanced dataset:

| Class      | Image Count | % of Dataset |
|------------|-------------|--------------|
| Airplane   | 510         | 19.6%        |
| Automobile | 515         | 19.8%        |
| Bird       | 527         | 20.2%        |
| Cat        | 518         | 19.9%        |
| Dog        | 535         | 20.5%        |
| **Total**  | **2,605**   | **100%**     |

That's right—I collected over 2,600 images across the five categories! And check out how balanced the dataset is—each class makes up almost 20% of the total. This balance is crucial for training unbiased models.

Here are some sample images I collected:

![Sample Images](/elec4630-blog/images/sample_images.png)

The variety is impressive: different angles, lighting conditions, backgrounds, and art styles. This diversity would help the model generalise new images better.

## Data Magic: Preprocessing and Augmentation

Getting good performance isn't just about having lots of data—it's about making the most of what you have. I used fastai's data augmentation tools to create variations of my training images:

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=parent_label,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
).dataloaders(path)
```

These augmentations apply random transformations like:
- 🔁 Random rotations and flips
- 🔍 Zooming in and out
- 🔆 Brightness and contrast adjustments
- 🖼️ Perspective warping

Here's a batch of augmented training images:

![Sample Batch](/elec4630-blog/images/sample_batch.png)

Look at how diverse these images are! Each time the model sees an image during training, it's slightly different. This forces the model to focus on the important features that define each class rather than memorising specific images.

My final dataset was split into:
- 📚 Training set: 2,084 images (80%)
- 🧪 Validation set: 521 images (20%)

## The Model: Standing on the Shoulders of Giants

For this project, I used transfer learning with a ResNet-34 model pretrained on ImageNet. This approach leverages knowledge from millions of pre-classified images, allowing my model to achieve impressive accuracy with just a small amount of training.

### Why ResNet-34?

ResNet-34 offers an excellent balance between accuracy and efficiency. With 34 layers and skip connections to prevent vanishing gradients, it's powerful enough to distinguish complex features and lightweight enough to train quickly.

### The Secret Sauce: Cross-Entropy Loss

For multi-class classification tasks like this one, cross-entropy loss is the perfect choice. In mathematical terms:

$\mathcal{L}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$

That might look intimidating, but the intuition is simple: the loss function strongly penalises the model when confident about a wrong answer. This pushes it to be both accurate and appropriately cautious in its predictions.

## Training Adventures

The training process was surprisingly fast! I used a two-stage approach:

1. First, I froze the pretrained layers and trained just the new classifier head for 3 epochs
2. Then I gradually unfroze the network and fine-tuned the entire model for 5 more epochs

```python
learn = vision_learner(
    dls,
    resnet34,
    metrics=[error_rate, accuracy],
    loss_func=CrossEntropyLossFlat()
)

# Find optimal learning rate
learn.lr_find()

# Train with fine-tuning
learn.fine_tune(5, freeze_epochs=3)
```

The results were impressive:

| Epoch | Training Loss | Validation Loss | Accuracy | Time |
|-------|--------------|----------------|----------|------|
| 0     | 1.487625     | 0.485106       | 83.30%   | 00:08|
| ...   | ...          | ...            | ...      | ...  |
| Final | 0.109311     | 0.321690       | 90.79%   | 00:10|

In just **93.74 seconds** of total training time, the model reached **90.79% accuracy**! That's remarkable efficiency—under 2 minutes to train a model that can distinguish between 5 different classes with high accuracy.

## Peeking Inside the Black Box: Confusion Matrix

One of my favourite parts of this project was analysing how the model makes decisions. A confusion matrix shows us which classes the model confuses with each other:

![Confusion Matrix](/elec4630-blog/images/confusion_matrix.png)

Looking at this visualisation, I discovered some fascinating patterns:

- The **strong diagonal** shows the model gets most predictions right
- The most common confusion? **Airplane ↔ Automobile** (probably because both are human-made vehicles)
- Surprisingly, the model rarely confused **cats and dogs** (only 4 cats classified as dogs)
- **Birds** were occasionally mistaken for **airplanes** (both fly in the sky!)

The classification report confirmed the model's strong performance:

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| airplane   | 0.89      | 0.91   | 0.90     |
| automobile | 0.91      | 0.93   | 0.92     |
| bird       | 0.90      | 0.91   | 0.91     |
| cat        | 0.94      | 0.91   | 0.92     |
| dog        | 0.91      | 0.88   | 0.89     |
| **overall**    | **0.91**     | **0.91**  | **0.91**   |

Even more impressive when you remember this model was trained in less than 2 minutes!

## t-SNE Visualisation for Feature Space Analysis

t-SNE (t-distributed Stochastic Neighbour Embedding) helps visualise high-dimensional data by projecting it into 2D space while preserving local relationships. I extracted 512-dimensional feature vectors from the penultimate layer of our model and applied t-SNE:

```python
# Extract features from model's body
body = learn.model[0]
features, labels = [], []

with torch.no_grad():
    for batch_idx, (xb, yb) in enumerate(learn.dls.valid):
        # Extract features
        feats = body(xb)
        # Apply global average pooling
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).view(feats.shape[0], -1)
        features.append(feats.cpu().numpy())
        labels.append(yb.cpu().numpy())
        
# Combine batches
features = np.concatenate(features)
labels = np.concatenate(labels)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_features = tsne.fit_transform(features)
```

The resulting visualisation reveals fascinating patterns in the feature space:

![t-SNE Visualisation](/elec4630-blog/images/tsne_visualisation.png)

### Feature Space Insights:

1. **Class Separation**:
   - All five classes form distinct clusters
   - Cat (teal) forms a tight cluster in the upper-right
   - Bird (green) is more dispersed, reflecting higher within-class variability
   - Dog (purple) clusters above the origin with moderate spread
   - Airplane (pink) sits below the origin in a compact group
   - Automobile (gold) occupies the lower-left quadrant

2. **Centroid Distances**:
   - **Closest pairs**:
     - Cat ↔ Dog: 15.19
     - Airplane ↔ Automobile: 16.79
     - Bird ↔ Cat: 17.52
   - **Most distant pairs**:
     - Automobile ↔ Bird: 42.55
     - Automobile ↔ Cat: 36.86
     - Bird ↔ Dog: 30.53

3. **Semantic Organization**:
   - Manufactured objects (airplane, automobile) cluster in the lower half
   - Animals separate primarily along the vertical axis
   - The model has learned meaningful semantic relationships, grouping similar classes closer together

4. **Correlation with Confusion Matrix**:
   - The closest centroids (cat-dog, airplane-automobile) align with our highest confusion rates
   - The most distant pairs (automobile-bird) correspond to minimal misclassifications

This visualisation confirms that the model's learned feature space captures semantic relationships between classes, explaining its successes and occasional failures.

## Testing on Sample Images

To qualitatively assess the model's performance, I tested it on clear, representative examples:

![Sample Predictions](/elec4630-blog/images/sample_predictions.png)

**Results**:
- **Perfect Classification**: 100% accuracy on these sample images
- **High Confidence**: Predictions had 0.9997-1.0000 probability for the correct class
- **Diverse Handling**: The model correctly classified different perspectives, lighting conditions, and compositions

For example, the airplane silhouette, front-facing automobile, colourful bird, accessorised cat, and relaxed dog were all classified with near-perfect confidence, demonstrating the model's robust generalisation.

## Challenges and Learning Points

Several valuable insights emerged during this project:

1. **Web Scraping Complexities**:
   - Search terms significantly impact dataset diversity
   - Multiple searches with varied terms yield better datasets than single broad searches
   - Image validation is crucial to remove corrupted downloads

2. **Transfer Learning Efficiency**:
   - Only 8 epochs were needed to reach 90%+ accuracy
   - Initial frozen training provides valuable feature extraction
   - Fine-tuning further adapts features to our specific classes

3. **t-SNE Interpretation**:
   - Perplexity value (30) affects cluster separation
   - Proximity in t-SNE space correlates with confusion likelihood
   - Classes with higher visual variability (birds) show greater spread in t-SNE space

## Conclusion

This multi-class image classification project demonstrated how quickly we can build effective computer vision models using:

1. **Web-scraped data** instead of standard datasets
2. **Transfer learning** for rapid training
3. **Advanced visualisation** for interpretability

The final model achieved an impressive **90.79% accuracy** across five diverse classes with minimal training time (93.74 seconds). The confusion matrix and t-SNE visualisation provided complementary insights into model performance and feature space organisation.

What's particularly fascinating is how the t-SNE visualisation revealed an organic organisation of classes that aligns with our intuitive understanding—grouping vehicles together and animals together, while still maintaining clear boundaries between classes.

This project demonstrated that modern deep learning tools and transfer learning approaches enable remarkably effective computer vision systems, even with modest computational resources and training time.

## References

1. Lovell, B. (2025). "Course22 repository." [https://github.com/lovellbrian/course22](https://github.com/lovellbrian/course22)
2. Lovell, B. (2025). "00-is-it-a-bird-creating-a-model-from-your-own-data.ipynb." Fast.ai Course Example.
3. Howard, J., & Gugger, S. (2020). *Deep Learning for Coders with fastai and PyTorch*. O'Reilly Media.
4. Van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE." *Journal of Machine Learning Research*, 9(Nov), 2579-2605.
5. Wattenberg, M., Viégas, F., & Johnson, I. (2016). "How to Use t-SNE Effectively." *Distill*. [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
6. ELEC4630 Assignment 2 (2025). "Question 4: Custom Image Classification." The University of Queensland.
