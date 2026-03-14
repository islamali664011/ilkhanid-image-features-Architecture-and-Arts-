import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import os
from extract_features import preprocess, model, device, image_folder, features_dict

def find_similar_image(query_image_path, features_dict, top_k=5):
    # Convert query image to feature vector
    img = Image.open(query_image_path).convert('RGB')
    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_feature = model(img_t).cpu().numpy()

    # Store similarity scores
    similarity_scores = []

    for filename, feature in features_dict.items():
        sim = cosine_similarity(query_feature, feature)[0][0]
        similarity_scores.append((filename, sim))

    # Sort by similarity
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Print top-k similar images
    print(f"Top {top_k} similar images:")
    for i, (filename, score) in enumerate(similarity_scores[:top_k]):
        print(f"{i+1}. {filename}  |  Similarity: {score:.4f}")

    # Plot query and similar images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, top_k+1, 1)
    plt.imshow(img)
    plt.title("Query Image")
    plt.axis('off')

    for i, (filename, score) in enumerate(similarity_scores[:top_k]):
        img_path = os.path.join(image_folder, filename)
        img_sim = Image.open(img_path).convert('RGB')
        plt.subplot(1, top_k+1, i+2)
        plt.imshow(img_sim)
        plt.title(f"{score:.2f}")
        plt.axis('off')

    plt.show()

# Example usage
query_image_path = '/content/BgXNqNoCcAEF5fx.jpeg'
find_similar_image(query_image_path, features_dict, top_k=5)
