import torch
import timm
from torchvision import transforms
from PIL import Image
import os

# Image folder
image_folder = '/content/ilkhanid_images'

# Load pretrained ViT model
model_name = 'vit_base_patch16_224'
model = timm.create_model(model_name, pretrained=True)
model.eval()
model.head = torch.nn.Identity()  # Remove classification head

# Preprocessing for images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std'])
])

# Dictionary to store features
features_dict = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Extract features for each image
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(image_folder, filename)
        img = Image.open(path).convert('RGB')
        img_t = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(img_t)

        features_dict[filename] = feature.cpu().numpy()

print(f"Extracted features for {len(features_dict)} images.")

# Save model weights
torch.save(model.state_dict(), '/content/vit_ilkhanid.pt')
