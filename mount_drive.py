from google.colab import drive
import shutil
import os

# Mount Google Drive
drive.mount('/content/drive')

# Source folder in Drive
source_folder = '/content/drive/MyDrive/project for ilkhanid'

# Destination folder in Colab
destination_folder = '/content/ilkhanid_images'

# Create folder if not exists
os.makedirs(destination_folder, exist_ok=True)

# Copy all images
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        shutil.copy(os.path.join(source_folder, filename),
                    os.path.join(destination_folder, filename))

print(f"Copied {len(os.listdir(destination_folder))} images to Colab!")
