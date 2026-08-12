# Smart Product Pricing 

Developed as part of a Amazon ML Challenge 2025 by Team Innovatrix

## Team Details

Team Name: Innovatrix  

Team Members:
- Sarthak Jain  
- Keshav Kumar  
- Aishwarya Lakshmi  
- Ritik Sharma  

---

## Overview

This project presents a machine learning solution for predicting product prices in an e-commerce environment. The model leverages both textual and visual data to learn complex relationships between product attributes and their corresponding prices.

The solution is based on a multimodal learning approach that combines natural language processing and computer vision techniques to improve prediction accuracy.

---

## Problem Statement

The objective is to predict the price of a product using:

- `catalog_content`: Product title, description, and item pack quantity  
- `image_link`: Product image  

The model must output a predicted price for each product in the test dataset.

---

## Dataset

Dataset:                                                                                                                                                           
https://drive.google.com/file/d/1C-dLcFeTHytMwhzuxAf8sLqVbakwyGCn/view?usp=drive_link


The dataset consists of:

- 75,000 training samples (with price labels)
- 75,000 test samples (without price labels)

### Features

- `sample_id`: Unique identifier  
- `catalog_content`: Textual product information  
- `image_link`: URL of product image  
- `price`: Target variable (training only)  

### Files

- `dataset/train.csv`
- `dataset/test.csv`
- `dataset/sample_test.csv`
- `dataset/sample_test_out.csv`

---

## Methodology

### Problem Analysis

Exploratory analysis showed that product price is influenced by:

- Brand presence  
- Product specifications  
- Quantity (IPQ)  
- Visual quality and branding in images  

The target variable exhibited skewness, which was handled using log transformation to stabilize training.

---

### Solution Approach

A hybrid multimodal pipeline was used to combine textual and visual features.

#### Text Processing

- Cleaning (removal of punctuation, HTML, special characters)
- Tokenization and lemmatization
- Embedding using BERT

#### Image Processing

- Image download using provided utilities
- Resizing to 224 × 224
- Feature extraction using ResNet50 (pretrained on ImageNet)

#### Feature Fusion

- Concatenation of BERT embeddings and image features
- Fully connected layers for regression output

---

## Model Architecture
catalog_content -> Text Preprocessing -> BERT Embeddings ----
-> Concatenation -> Dense Layers -> Price
image_link -> Image Processing -> ResNet50 Features --


---

## Evaluation Metric

The model is evaluated using Symmetric Mean Absolute Percentage Error (SMAPE):

SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|) / 2)

- Range: 0% to 200%  
- Lower values indicate better performance  

---

## Model Performance

- SMAPE: 12.47  
- MAE: 5.2  
- RMSE: 7.4  
- R² Score: 0.89  

The multimodal model outperformed text-only approaches, demonstrating the importance of combining image and textual features.

---

## Output Format

The final submission must be a CSV file:

sample_id,price  
12345,199.99  
12346,349.50  

Requirements:

- All sample IDs must be included  
- Prices must be positive float values  
- Format must exactly match `sample_test_out.csv`  

---

