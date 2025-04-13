#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Devanagari Handwritten Character Recognition
Training Script

This script trains a Convolutional Neural Network to recognize
handwritten Devanagari characters and digits. It handles data loading, 
preprocessing, model definition, training, and saving.

"""

# Import essential libraries
import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configure TensorFlow
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Create a concrete mapping between folder names and actual Devanagari characters
devanagari_char_map = {
    'character_1_ka': 'क', 'character_2_kha': 'ख', 'character_3_ga': 'ग', 'character_4_gha': 'घ', 'character_5_kna': 'ङ',
    'character_6_cha': 'च', 'character_7_chha': 'छ', 'character_8_ja': 'ज', 'character_9_jha': 'झ', 'character_10_yna': 'ञ',
    'character_11_taamatar': 'ट', 'character_12_thaa': 'ठ', 'character_13_daa': 'ड', 'character_14_dhaa': 'ढ', 'character_15_adna': 'ण',
    'character_16_tabala': 'त', 'character_17_tha': 'थ', 'character_18_da': 'द', 'character_19_dha': 'ध', 'character_20_na': 'न',
    'character_21_pa': 'प', 'character_22_pha': 'फ', 'character_23_ba': 'ब', 'character_24_bha': 'भ', 'character_25_ma': 'म',
    'character_26_yaw': 'य', 'character_27_ra': 'र', 'character_28_la': 'ल', 'character_29_waw': 'व', 'character_30_motosaw': 'श',
    'character_31_petchiryakha': 'ष', 'character_32_patalosaw': 'स', 'character_33_ha': 'ह', 'character_34_chhya': 'क्ष', 'character_35_tra': 'त्र',
    'character_36_gya': 'ज्ञ', 'digit_0': '०', 'digit_1': '१', 'digit_2': '२', 'digit_3': '३',
    'digit_4': '४', 'digit_5': '५', 'digit_6': '६', 'digit_7': '७', 'digit_8': '८', 'digit_9': '९'
}

def create_character_mappings(data_dir):
    """
    Create mappings between class indices, folder names, and Devanagari characters.
    
    Args:
        data_dir: Path to the training data directory
        
    Returns:
        idx_to_char: Dictionary mapping class indices to Devanagari characters
        char_to_idx: Dictionary mapping Devanagari characters to class indices
        classes: List of class folder names
    """
    # Use sorted to ensure consistent ordering of classes
    classes = sorted(os.listdir(data_dir))
    
    # Create index to character mapping
    idx_to_char = {}
    char_to_idx = {}
    
    for idx, class_name in enumerate(classes):
        char = devanagari_char_map.get(class_name, class_name)
        idx_to_char[str(idx)] = char  # Convert keys to strings for JSON compatibility
        char_to_idx[char] = idx
        print(f"{idx}: {class_name} -> '{char}'")
    
    # Save the mappings to a JSON file
    with open('devanagari_mappings.json', 'w', encoding='utf-8') as f:
        json.dump({
            'idx_to_char': idx_to_char,
            'char_to_idx': char_to_idx,
            'class_names': classes
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Saved character mappings to devanagari_mappings.json")
    
    return idx_to_char, char_to_idx, classes

def load_and_preprocess_data(data_dir, classes):
    """
    Load and preprocess the training data.
    
    Args:
        data_dir: Path to the training data directory
        classes: List of class folder names
        
    Returns:
        X: Preprocessed image data (normalized and reshaped)
        y: Class labels
    """
    # Initialize lists to store data and labels
    X = []
    y = []
    
    # Load images and labels
    for idx, class_name in enumerate(tqdm(classes, desc="Loading data")):
        class_dir = os.path.join(data_dir, class_name)
        char = devanagari_char_map.get(class_name, class_name)
    
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize all images to a standard size
                img = cv2.resize(img, (32, 32))
                X.append(img)
                y.append(idx)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Normalize pixel values to be between 0 and 1
    X = X / 255.0
    
    # Reshape the data for CNN input
    X = X.reshape(-1, 32, 32, 1)
    
    return X, y

def visualize_sample_data(X_train, y_train, idx_to_char):
    """
    Visualize sample images from the training data.
    
    Args:
        X_train: Training image data
        y_train: Training labels (one-hot encoded)
        idx_to_char: Mapping from class indices to Devanagari characters
    """
    plt.figure(figsize=(10, 5))
    for i in range(10):
        class_idx = np.argmax(y_train[i]) if len(y_train.shape) > 1 else y_train[i]
        char = idx_to_char[str(class_idx)]
    
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i].reshape(32, 32), cmap='gray')
        plt.title(f"{char} (class {class_idx})")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_training_data.png')
    plt.show()

def build_model(input_shape, num_classes):
    """
    Build and compile the CNN model.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    """
    Train the model and return training history.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        history: Training history object
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Training history object
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def save_model_and_metadata(model, classes, idx_to_char):
    """
    Save the model in TF and Keras formats, and create metadata file.
    
    Args:
        model: Trained Keras model
        classes: List of class names
        idx_to_char: Mapping from class indices to Devanagari characters
    """
    # Save the model in TF format (recommended for TensorFlow.js conversion)
    model.save('devanagari_character_recognition')
    
    # Save in Keras format if needed
    model.save('devanagari_character_recognition.keras')
    
    # Create a model metadata file that includes the character mappings
    with open('model_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({
            'num_classes': len(classes),
            'char_mappings': idx_to_char,
            'input_shape': [32, 32, 1],
            'class_names': classes
        }, f, ensure_ascii=False, indent=2)
    
    print("Model training completed and saved successfully!")
    
    # Print TensorFlow.js conversion command
    print("\nTo convert the model for TensorFlow.js, run the following command:")
    print("tensorflowjs_converter --input_format=tf_saved_model devanagari_character_recognition devanagari_model_js")

def main():
    """Main function to run the training pipeline."""
    # Define paths
    data_dir = 'DevanagariHandwrittenCharacterDataset/Train'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please download and extract the Devanagari Handwritten Character Dataset.")
        return
    
    # Create character mappings
    idx_to_char, char_to_idx, classes = create_character_mappings(data_dir)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(data_dir, classes)
    
    # Convert labels to one-hot encoding
    y_one_hot = to_categorical(y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    
    # Display some sample images
    visualize_sample_data(X_train, y_train, idx_to_char)
    
    # Print data shapes
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Number of classes: {y_one_hot.shape[1]}")
    
    # Build and display model
    model = build_model(input_shape=(32, 32, 1), num_classes=len(classes))
    model.summary()
    
    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model and metadata
    save_model_and_metadata(model, classes, idx_to_char)

if __name__ == "__main__":
    main()