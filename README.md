# NeuralMood
## Overview

This project is an offline application for emotion detection and sentiment analysis using deep learning. It allows users to analyze images and classify emotions using a trained CNN model. Users can either analyze predefined datasets or upload their own images for analysis.

## Features

- Train and test a convolutional neural network (CNN) for emotion detection
- Load a pre-trained model if available
- Perform sentiment analysis on a test dataset
- Drag and drop images for custom analysis
- GUI-based application for ease of use

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- TensorFlow
- NumPy
- PIL (Pillow)
- Tkinter
- tkinterdnd2

### Install Dependencies

Run the following command to install required packages:

```sh
pip install tensorflow numpy pillow tkinterdnd2
```

## Dataset Structure

The dataset should be structured as follows:

```
C:\internshhip\emotion detection and sentimental analysis\
|-- train\
    |-- happy\
    |-- sad\
    |-- neutral\
|-- test\
    |-- happy\
    |-- sad\
    |-- neutral\
```

## Usage

### 1. Train the Model

If no pre-trained model exists, the script will automatically train the model.

```sh
python train_model.py
```

### 2. Run the Application

To start the GUI application:

```sh
python app.py
```

## How to Use the Application

1. **View Model Accuracy**: Displays the model's sentiment analysis results.
2. **Analyze Custom Images**:
   - Click on "Analyze Your Own Set of Images" to open a file dialog.
   - Drag and drop images into the designated area.
   - The application will display sentiment analysis results.

## Model Details

- CNN model with three convolutional layers
- Batch normalization and max pooling for improved accuracy
- Dropout layers to prevent overfitting
- Softmax activation for multi-class classification

## Troubleshooting

- If you encounter a missing module error, install the required dependencies.
- Ensure your dataset is structured correctly.
- If drag-and-drop is not working, reinstall `tkinterdnd2`.

## Future Enhancements

- Add real-time webcam-based emotion detection
- Improve accuracy with a larger dataset
- Optimize the model for faster inference

## AUTHOR 

DHRUVA KASHYAP
