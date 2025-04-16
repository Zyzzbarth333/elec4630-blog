---
layout: post
title: "Building a Web-Based Fingerprint Recognition System"
date: 2025-04-15 18:00:00 +1000
categories: [biometrics, computer-vision, python]
tags: [fingerprints, flask, opencv, recognition, gui]
toc: true
comments: true
image: images/fingerprint/fingerprint_header.jpg
description: "My journey implementing a web-based fingerprint recognition system with Python, OpenCV, and Flask, including enrollment, matching, and performance evaluation with ROC curves."
---

# Building a Web-Based Fingerprint Recognition System

Biometric identification systems, particularly fingerprint recognition, have become increasingly prevalent in our daily lives. From unlocking smartphones to accessing secure facilities, these systems offer a unique blend of security and convenience. In this post, I'll share my experience developing a fingerprint recognition system using Python, OpenCV, and Flask.

## Project Overview

The project required implementing a complete fingerprint recognition system with:

- A user-friendly interface for enrolling and verifying fingerprints
- A database to store fingerprint templates
- Performance evaluation tools including ROC curves
- Threshold optimisation for minimising error rates

The implementation is based on Minutiae-based matching, one of the most widely used techniques in fingerprint recognition. Let's explore the development process, challenges encountered, and solutions developed.

## Environment Setup and Initial Challenges

My journey began with an analysis of the provided Jupyter notebook from Professor Lovell's GitHub repository, which contained essential fingerprint processing steps including segmentation, orientation estimation, enhancement, and minutiae detection.

The first challenge emerged immediately: the notebook required OpenCV with the contrib modules for critical operations like thinning (skeletonisation). Attempting to run the code resulted in this error:

```python
AttributeError: module 'cv2' has no attribute 'ximgproc'
```

This was resolved by installing the OpenCV contrib package:

```bash
pip install opencv-contrib-python
```

## Architecturing a Modular Solution

After successfully running the notebook, I refactored the code into a modular architecture with clear separation of concerns:

```
fingerprint_system/
├── fingerprint_processor.py   # Core algorithms
├── utils.py                   # Helper functions
├── flask_app.py               # Web interface
├── templates/                 # HTML templates
│   ├── index.html
│   ├── enroll.html
│   ├── verify.html
│   └── evaluate.html
└── static/                    # CSS, JS, images
    └── ...
```

The `FingerprintProcessor` class encapsulates the complete processing pipeline:

```python
class FingerprintProcessor:
    def __init__(self, block_size=16, threshold=0.1):
        self.block_size = block_size
        self.threshold = threshold
        
    def process(self, image):
        # 1. Normalise and enhance the image
        normalized = self._normalize(image)
        
        # 2. Estimate ridge orientation
        orientation = self._estimate_orientation(normalized)
        
        # 3. Extract fingerprint region (segmentation)
        mask = self._segment(normalized, orientation)
        
        # 4. Enhance using Gabor filtering
        enhanced = self._enhance(normalized, orientation, mask)
        
        # 5. Extract minutiae features
        minutiae = self._extract_minutiae(enhanced)
        
        return {
            'enhanced': enhanced,
            'orientation': orientation,
            'mask': mask,
            'minutiae': minutiae
        }
        
    # Private methods for each processing step...
```

## Web-Based GUI: From Tkinter to Flask

Initially, I attempted to develop a traditional desktop GUI using Tkinter, but this approach faced significant challenges when running in containerised environments:

```python
# This approach didn't work well with Docker
import tkinter as tk
root = tk.Tk()
# ModuleNotFoundError: No module named 'tkinter'
```

After evaluating alternatives, I pivoted to a web-based approach using Flask, which offered several advantages:

1. Compatibility with containerised environments
2. Enhanced accessibility from any device with a web browser
3. Modern UI capabilities through HTML, CSS, and JavaScript
4. Simplified deployment options

The Flask application follows the Model-View-Controller (MVC) pattern:

```python
from flask import Flask, render_template, request, jsonify
import os
import sys
import cv2
import numpy as np

# Add module directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from fingerprint_processor import FingerprintProcessor

app = Flask(__name__)
processor = FingerprintProcessor()
database = {}  # Simple in-memory database for templates

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enroll', methods=['GET', 'POST'])
def enroll():
    if request.method == 'GET':
        return render_template('enroll.html')
    
    # POST processing for enrollment
    image_file = request.files['fingerprint']
    name = request.form['name']
    
    # Process image and store template
    image = cv2.imdecode(
        np.frombuffer(image_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    
    features = processor.process(image)
    database[name] = features['minutiae']
    
    return jsonify({'status': 'success', 'message': f'Enrolled {name}'})

# Routes for verification and evaluation...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

## Package Structure and Import Challenges

The modular architecture created an unexpected challenge with Python's import system:

```
ModuleNotFoundError: No module named 'utils'
```

This required proper Python package structuring with `__init__.py` files and absolute imports:

```python
# Before (problematic)
import utils

# After (working)
from fingerprint_recognition import utils
```

Additionally, for the web application to find modules correctly, I needed to explicitly add the module directory to Python's path:

```python
current_dir = os.path.dirname(os.path.abspath(__file__))
fingerprint_dir = os.path.join(current_dir, 'fingerprint_recognition')
sys.path.append(fingerprint_dir)
```

## Enhanced User Interface

The web interface features a responsive design with Bootstrap and custom JavaScript for real-time feedback. The UI includes:

- Drag-and-drop file uploads
- Interactive sliders for threshold adjustment
- Visual feedback with colour-coded matching results
- Comprehensive performance visualisations

## Dealing with Real-World Data: FVC2000 DB1 Integration

To thoroughly evaluate the system, I integrated the FVC2000 DB1 fingerprint dataset. This dataset follows a specific naming convention:

```
XXX_Y.tif
```

Where `XXX` represents the person ID and `Y` denotes the finger ID.

Working with this structured dataset presented several challenges, including Windows/Linux path resolution issues:

```
Error: DB1 path not found: \\wsl.localhost\Ubuntu\home\Assignment 2 -
Deep Learning and Biometrics\data\DB1_B
```

The solution was to develop a dedicated import module that intelligently parses the dataset structure and supports different partitioning strategies:

```python
def import_db1_dataset(path, strategy='first-impression'):
    """
    Import FVC2000 DB1 dataset and partition according to strategy.
    
    Parameters:
    -----------
    path : str
        Path to DB1 dataset directory
    strategy : str
        Partitioning strategy:
        - 'first-impression': First impression to gallery, rest to probe
        - 'person-split': Some subjects to gallery, others to probe
        - 'random-split': Random assignment based on configurable ratio
    
    Returns:
    --------
    dict
        Dictionary with 'gallery' and 'probe' keys
    """
    # Implementation...
```

## Performance Evaluation and ROC Analysis

The system includes a comprehensive performance evaluation module that:

1. Computes similarity scores for all genuine and impostor comparisons
2. Generates the ROC curve showing the trade-off between false accepts and false rejects
3. Automatically determines the threshold required for a 1% False Negative Rate
4. Visualises confusion matrices and score distributions

The ROC curve analysis revealed an Equal Error Rate (EER) of approximately 5-10%, with a False Positive Rate of 10-15% when operating at a 1% False Negative Rate.

```python
def calculate_roc(genuine_scores, impostor_scores):
    """
    Calculate ROC curve points and statistics.
    
    Parameters:
    -----------
    genuine_scores : list
        Similarity scores from genuine comparisons (same finger)
    impostor_scores : list
        Similarity scores from impostor comparisons (different fingers)
    
    Returns:
    --------
    dict
        Dictionary with FPR, TPR, thresholds, and EER
    """
    # Combined scores and ground truth labels
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # Calculate EER
    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]
    
    # Find threshold for 1% FNR
    fnr_target = 0.01
    fnr_index = np.argmin(np.abs(fnr - fnr_target))
    fpr_at_target = fpr[fnr_index]
    threshold_at_target = thresholds[fnr_index]
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'eer': eer,
        'fpr_at_target_fnr': fpr_at_target,
        'threshold_at_target_fnr': threshold_at_target
    }
```

## Advanced Matching Visualisation

A key enhancement was the implementation of detailed match visualisation to improve system interpretability:

```python
def visualize_match(query_image, gallery_image, matched_minutiae, score):
    """
    Generate a visual representation of matching minutiae between two
    fingerprint images.
    
    Parameters:
    -----------
    query_image : ndarray
        Query fingerprint image
    gallery_image : ndarray
        Gallery fingerprint image
    matched_minutiae : list
        Pairs of matching minutiae indices
    score : float
        Similarity score
        
    Returns:
    --------
    ndarray
        Visualization image with connected matching minutiae
    """
    # Implementation...
```

This visualisation helps in understanding why certain fingerprints match or don't match, and provides insights for improving the algorithm.

## Challenges and Lessons Learned

Throughout this project, several valuable lessons emerged:

1. **Environment Management**: Containerisation (Docker) provides consistency but creates unique challenges for GUI development
2. **Web vs Desktop**: Web interfaces offer significant advantages for accessibility and deployment
3. **Algorithm Tuning**: Fingerprint recognition is highly sensitive to parameter tuning
4. **Error Analysis**: Visualising matching points is crucial for understanding system behaviour
5. **Performance Trade-offs**: Security (low false accepts) must be balanced with convenience (low false rejects)

## Future Improvements

The current implementation successfully addresses all core requirements, but several areas for future improvement have been identified:

1. **Feature Enhancement**: Implementing more sophisticated minutiae descriptors
2. **Deep Learning Integration**: Exploring CNN-based approaches for feature extraction
3. **Performance Optimisation**: Improving processing speed for real-time applications
4. **Multi-modal Integration**: Combining fingerprints with other biometrics
5. **Spoof Detection**: Adding liveness detection to prevent presentation attacks

## Conclusion

Developing this fingerprint recognition system provided hands-on experience with computer vision techniques, web application development, and performance evaluation methodologies. The web-based approach proved to be a flexible and practical solution for biometric system implementation.

The modular architecture allows for easy extension and improvement, while the comprehensive evaluation tools provide insights into system performance. The experience gained through this project has deepened my understanding of both the theoretical and practical aspects of biometric recognition systems.

What other biometric modalities would you like to see explored in future posts? Let me know in the comments below!

---

## References

1. Maltoni, D., Maio, D., Jain, A.K., & Prabhakar, S. (2009). Handbook of Fingerprint Recognition. Springer Science & Business Media.
2. FVC2000 Fingerprint Database: [http://bias.csr.unibo.it/fvc2000/](http://bias.csr.unibo.it/fvc2000/)
3. Lovell, B. (2025). Fingerprint Recognition Notebook: [https://github.com/lovellbrian/fingerprint](https://github.com/lovellbrian/fingerprint)
4. Flask Web Framework: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
5. OpenCV Computer Vision Library: [https://opencv.org/](https://opencv.org/)

## Footnotes

[^1]: This is the footnote.

