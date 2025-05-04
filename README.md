# ColorFusionMetrics

## Overview

This project demonstrates the process of image colorization using deep learning models in PyTorch. The model is trained on the CIFAR-10 dataset and accepts grayscale images as input, producing plausible RGB colorized outputs. The training is tested with different loss functions, including:
- Mean Squared Error (MSE) Loss
- L1 Loss (MAE)
- Perceptual Loss (based on VGG16 features)

## Features

- Uses the CIFAR-10 dataset (tiny, but useful for experimentation)
- Converts RGB to grayscale as input
- Outputs full-color images from grayscale
- Compares training with different loss functions
- Includes perceptual loss with VGG16 feature maps
- Image post-processing with HSV exaggeration for enhanced saturation
- Supports visualization of original, grayscale, and colorized images
- Handles custom image uploads for inference

## Loss Functions

You can experiment with the following loss functions:

1. nn.MSELoss() - Penalizes larger errors more severely.

2. nn.L1Loss() - More robust to outliers.

3. PerceptualLoss() - Based on VGG16 feature similarity for more human-perceived realism.

- Switch between losses by assigning loss_function = ... in the training script.

## Tech Stack

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- PIL (Pillow)
- numpy

<!--start code-->
### Install dependencies:

    pip install torch torchvision matplotlib pillow numpy

<!--end code-->

## Experiments and Observations

Loss Function | Visual Quality | Training Time | Notes
MSE Loss | Moderate | Fast | Over-smooth results
L1 Loss | Better detail | Fast | Sharper edges
Perceptual | Best looking | Slower | More realistic colors but longer training
