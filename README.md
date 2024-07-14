# Conditional GAN for Image Generation from Text Descriptions

This project implements a Conditional Generative Adversarial Network (GAN) that generates images from textual descriptions. The generator takes both noise and text embeddings as inputs to produce images, while the discriminator evaluates the authenticity of the generated images.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchvision
- spaCy
- PIL (Python Imaging Library)
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/conditional-gan-text2image.git
    ```

2. Install the required libraries:
    ```bash
    pip install torch torchvision spacy pillow numpy matplotlib
    python -m spacy download en_core_web_md
    ```

## Dataset Preparation

Prepare your dataset with two directories:
- `root_image_dir`: Directory containing image files.
- `root_text_dir`: Directory containing text files with descriptions corresponding to the images.

## Model Architecture

### Conditional Generator
The generator uses a combination of noise and text embeddings to generate images.

### Discriminator
The discriminator evaluates the generated images against real images.

## Acknowledgments

Inspired by various implementations of GANs and conditional GANs available online.
