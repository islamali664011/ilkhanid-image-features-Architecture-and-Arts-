# ilkhanid-image-features-Architecture-and-Arts-
To extract the complete characteristics of Ilkhanid architecture and art in Iran, and then compare them with their counterparts in the Islamic world

# Ilkhanid Image Feature Extraction

## Description
This project extracts features from images of Ilkhanid architecture and art using a pretrained Vision Transformer (ViT) model. The extracted features can be used to compare images and find similarities using Cosine Similarity.

---

## Requirements
- Python >= 3.7  
- Libraries:
  - torch
  - timm
  - torchvision
  - Pillow
  - numpy
  - scikit-learn
  - matplotlib

Install all dependencies with:
```bash
pip install -r requirements.txt

## Usage
Prepare your image folder on Google Drive or locally.

Update the paths in the code (source_folder and query_image_path) as needed.

Run the script to extract features for all images.

Use the find_similar_image() function to find the most similar images to a query image.


## Files

ilkhanid_feature_extraction.py: Main script for feature extraction and image comparison.
requirements.txt: List of required Python libraries.

## Notes

The model uses GPU if available to speed up feature extraction.
The project can be extended for image classification or detecting architectural patterns.

## Author
Islam Ali Mohamed
Specialist in Islamic architecture and art, and AI-based image analysis.



