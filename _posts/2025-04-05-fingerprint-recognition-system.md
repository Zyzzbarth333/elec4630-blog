---
layout: post
title: "Building a Web-Based Fingerprint Recognition System"
date: 2025-04-05 18:00:00 +1000
categories: [biometrics, computer-vision, python]
tags: [fingerprints, flask, opencv, recognition, gui]
toc: true
comments: true
image: images/fingerprint/fingerprint_header.jpg
description: "End-to-end fingerprint enrolment, verification and ROC evaluation in Python + Flask—developed for ELEC4630 Assignment 2."
---

# Building a Web-Based Fingerprint Recognition System

This post documents the system I built for **Assignment 2 – Question 1**: a self-contained app that

* enrols fingerprints,
* matches probes against a gallery,
* evaluates performance on the FVC2000 DB1_B dataset, and
* reports ROC-curve metrics including the False-Positive Rate at 1 % FNR.

---

## How the System Meets Each Requirement

| Assignment 2 Q1 task | Where it happens | Screenshot / figure |
|----------------------|------------------|---------------------|
| Enrol a fingerprint and name | `index.html` → `/enroll` route | Fig 1 |
| Compare a probe to gallery   | `verify.html` → `/verify` | Fig 2 |
| Bulk evaluation on many prints | `performance_analyser.py` | Fig 3 (ROC) |
| Generate ROC curve           | `calculate_roc()`          | Fig 3 |
| Report FPR when FNR = 1 %    | Auto-threshold in `performance_analyser.py` | Table 1 |

---

## Project Directory (key folders)

```text
/workspaces/fingerprint/
├── flask_app.py             # Flask entry-point
├── data/
│   ├── database/            # SQLite DB + Pickle backup
│   ├── DB1_B/               # Evaluation prints
│   └── samples/             # Demo images
├── app/
│   ├── core/                # Algorithms
│   │   ├── fingerprint_processor.py
│   │   └── performance_analyser.py
│   ├── utils/
│   │   ├── utils.py
│   │   └── db_manager.py
│   ├── static/results/      # Generated plots
│   └── templates/           # Jinja2 pages
```

SQLite is the default persistent store; a Pickle snapshot makes quick offline demos easy.

---

## Key Algorithm Steps

1. **Enhancement** – Gabor filter bank boosts ridge contrast.  
2. **Skeletonisation** – Guo-Hall thinning (`cv.ximgproc.thinning`).  
3. **Minutiae detection** – ridge endings & bifurcations extracted.  
4. **Matching** – polar alignment + RANSAC choose correspondences.  
5. **Scoring** – similarity normalised to 0–1.

---

## Evaluation on FVC2000 DB1_B

### ROC Curve

![ROC curve](/app/static/results/roc_db1b.png)

### Summary Metrics (Table 1)

| Metric | Result |
|--------|--------|
| Equal Error Rate | ≈ 7 % |
| FPR @ 1 % FNR   | 12.3 % |
| Threshold @ 1 % FNR | 0.69 |

---

## Visualising a Successful Match

![Matched minutiae overlay](images/fingerprint/test_verification_success.jpg)

Colour-coded lines connect matched minutiae—useful for quick sanity checks.

---

## Lessons Learned

* **Web beats desktop** in containerised environments—Flask worked everywhere Tkinter didn’t.  
* **Parameter tuning matters**—ridge-frequency and binarisation thresholds have a big impact on EER.  
* **Security vs convenience**—tightening the threshold lowers false accepts but increases false rejects.

---

## References

1. Maltoni, D. et al. *Handbook of Fingerprint Recognition* (2009).  
2. FVC2000 Fingerprint Database – DB1_B.  
3. Lovell, B., “Fingerprint Recognition Notebook,” GitHub.  
4. Flask Web Framework – <https://flask.palletsprojects.com/>  
5. OpenCV Library – <https://opencv.org/>
```
