#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Devanagari Handwritten Character Recognition
Test Script

This script evaluates the trained CNN model on the test dataset,
visualizes predictions, and generates a confusion matrix.

"""

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Path to test data directory
TEST_DATA_DIR = 'DevanagariHandwrittenCharacterDataset/Test'

def preprocess_image(img_path):
    """
    Load and preprocess a single image.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Preprocessed image or None if loading fails
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {img_path}")
        return None
    
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 1)
    return img

def display_predictions(images, true_labels, pred_labels, confidences, num_images=10):
    """
    Visualize model predictions with color-coded correctness.
    
    Args:
        images: Array of test images
        true_labels: Array of true class labels
        pred_labels: Array of predicted class labels
        confidences: Array of prediction confidence scores
        num_images: Number of images to display
    """
    n = min(len(images), num_images)
    fig, axes = plt.subplots(2, n//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n):
        img = images[i].reshape(32, 32)
        axes[i].imshow(img, cmap='gray')
        
        # Color coding: green for correct, red for incorrect predictions
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        title = f"True: Class {true_labels[i]}\nPred: Class {pred_labels[i]}\nConf: {confidences[i]:.2f}"
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels, class_labels):
    """
    Generate and plot a confusion matrix.
    
    Args:
        true_labels: Array of true class labels
        pred_labels: Array of predicted class labels
        class_labels: List of class label strings
    """
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, pred_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def test_random_noise(model, metadata):
    """
    Test the model with random noise when no test data is available.
    
    Args:
        model: Loaded Keras model
        metadata: Model metadata including character mappings
    """
    print("Testing model with random noise...")
    
    # Get character mappings
    idx_to_char = metadata['char_mappings']
    num_classes = len(idx_to_char)
    
    # Generate 10 random noise images
    test_images = []
    for i in range(10):
        # Create random noise
        noise = np.random.rand(32, 32, 1)
        test_images.append(noise)
    
    test_images = np.array(test_images)
    
    # Make predictions
    predictions = model.predict(test_images)
    pred_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Display predictions on random noise
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[i].reshape(32, 32), cmap='gray')
        plt.title(f"Pred: Class {pred_classes[i]}\nConf: {confidences[i]:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nNote: These are random noise images, so predictions are meaningless.")
    print("The model is forced to classify them into one of the known classes.")

def test_single_image(img_path, model, metadata):
    """
    Test the model on a single image.
    
    Args:
        img_path: Path to the image file
        model: Loaded Keras model
        metadata: Model metadata including character mappings
    
    Returns:
        Tuple of (predicted class index, confidence score)
    """
    # Preprocess the image
    img = preprocess_image(img_path)
    if img is None:
        return None, 0
    
    # Make prediction
    pred = model.predict(img)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    
    # Display the image and prediction
    plt.figure(figsize=(6, 6))
    plt.imshow(img.reshape(32, 32), cmap='gray')
    plt.title(f"Predicted: Class {class_idx}\nConfidence: {confidence:.4f}")
    plt.axis('off')
    plt.show()
    
    return class_idx, confidence

def main():
    """Main function to test the model."""
    print("Testing Devanagari Character Recognition Model")
    
    # Load the model
    try:
        model = load_model('devanagari_character_recognition.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load character mappings
    try:
        with open('model_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print("Metadata loaded successfully!")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Check if test directory exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Test directory not found: {TEST_DATA_DIR}")
        # Instead, let's create some synthetic test data from random noise
        print("Creating synthetic test data...")
        test_random_noise(model, metadata)
        return
    
    # Get class names and mappings
    class_names = metadata['class_names']
    idx_to_char = metadata['char_mappings']
    
    # Load a batch of test images
    test_images = []
    true_labels = []
    class_images = {}
    
    # Try to get a few images from each class
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        
        img_files = os.listdir(class_dir)[:3]  # Take up to 3 images per class
        for img_file in img_files:
            img_path = os.path.join(class_dir, img_file)
            img = preprocess_image(img_path)
            if img is not None:
                test_images.append(img[0])  # Remove batch dimension
                true_labels.append(class_idx)
                
                if class_name not in class_images:
                    class_images[class_name] = []
                if len(class_images[class_name]) < 5:  # Limit to 5 images per class
                    class_images[class_name].append((img[0], class_idx))
    
    if not test_images:
        print("No test images found. Creating synthetic test data...")
        test_random_noise(model, metadata)
        return
        
    # Convert to numpy arrays
    test_images = np.array(test_images)
    test_images = test_images.reshape(-1, 32, 32, 1)
    true_labels = np.array(true_labels)
    
    # Make predictions
    predictions = model.predict(test_images)
    pred_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(true_labels) == pred_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Display some predictions
    display_predictions(test_images, true_labels, pred_classes, confidences)
    
    # Create class label strings for the confusion matrix
    class_label_strings = [f"Class {i}" for i in range(len(class_names))]
    
    # If there are enough samples, show confusion matrix for a subset of classes
    if len(set(true_labels)) > 5 and len(true_labels) > 20:
        # Take top 10 most frequent classes
        unique_classes, counts = np.unique(true_labels, return_counts=True)
        top_classes_idx = np.argsort(counts)[-10:]
        
        # Filter data to only include these classes
        mask = np.isin(true_labels, top_classes_idx)
        if np.sum(mask) > 10:  # Ensure we have enough samples
            filtered_true = true_labels[mask]
            filtered_pred = pred_classes[mask]
            
            # Get subset of class labels for the confusion matrix
            filtered_class_labels = [class_label_strings[i] for i in top_classes_idx]
            
            # Plot confusion matrix
            plot_confusion_matrix(filtered_true, filtered_pred, filtered_class_labels)
    
    # Show per-class accuracy
    print("\nPer-class accuracy:")
    for class_name in sorted(class_images.keys())[:10]:  # Show first 10 classes
        samples = class_images[class_name]
        if not samples:
            continue
            
        images = np.array([sample[0] for sample in samples]).reshape(-1, 32, 32, 1)
        true_class_idx = -1
        
        # Find the class index for this class name
        for idx, name in enumerate(class_names):
            if name == class_name:
                true_class_idx = idx
                break
                
        if true_class_idx == -1:
            continue
            
        preds = model.predict(images)
        pred_classes = np.argmax(preds, axis=1)
        
        accuracy = sum(1 for idx in pred_classes if idx == true_class_idx) / len(pred_classes)
        print(f"{class_name} (Class {true_class_idx}): {accuracy:.2f}")

if __name__ == "__main__":
    main()