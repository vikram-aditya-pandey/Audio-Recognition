🔊 Audio Recognition with MFCC and CNN

This project implements an audio classification system that uses **MFCC** 
features extracted from sound files and trains a **Convolutional Neural Network (CNN)** 
to recognize different audio classes.

🚀 Features

* Extracts MFCC features from audio clips
* Builds and trains a CNN model using TensorFlow/Keras
* Supports multi-class audio classification
* Visualizes training metrics (loss and accuracy)
* Evaluates model performance on test data

🧠 Technologies Used

* `Python`
* `librosa` – for audio processing and MFCC extraction
* `numpy`, `pandas` – for data handling
* `matplotlib`, `seaborn` – for visualization
* `tensorflow` / `keras` – for building and training the CNN


## 🛠 How It Works

1. **Preprocessing**:

   * Loads `.wav` files from class-labeled directories.
   * Extracts MFCC features using `librosa`.
   * Pads/truncates each MFCC to a consistent shape.

2. **Dataset Creation**:

   * Stores MFCCs and labels into NumPy arrays.
   * Splits data into training and testing sets.

3. **Model Architecture**:

   * A simple CNN with:

     * Convolutional layers
     * MaxPooling
     * Dropout for regularization
     * Dense layers with softmax for classification

4. **Training**:

   * Compiles and trains the CNN on the MFCC dataset.
   * Uses early stopping or validation split if desired.

5. **Evaluation**:

   * Tests the model on unseen data.
   * Plots training/validation accuracy and loss.


## 🧠 Model Summary

Example architecture:

```text
Input: (40 MFCCs x N frames)

Conv2D -> ReLU -> MaxPooling
Conv2D -> ReLU -> MaxPooling
Flatten
Dense -> Dropout
Dense (softmax)
```

⚠️ Notes

* Audio files should be consistent in format (e.g., `.wav`, mono, same sampling rate).
* Padding/truncation ensures fixed input size to the CNN.
* More complex architectures (like CRNNs or transformer-based models) can improve accuracy on larger datasets.
