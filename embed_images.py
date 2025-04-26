import os
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from docarray import BaseDoc, DocList
from docarray.typing import TorchTensor
from docarray.utils.find import find

# Define the custom document schema
class RestaurantImageDoc(BaseDoc):
    uri: str
    embedding: TorchTensor
    restaurant_name: str

# Load Jina CLIP model with Auto classes
processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Image folder
data_dir = "restaurant_images"
docs = DocList[RestaurantImageDoc]()

# Embed images
for restaurant in os.listdir(data_dir):
    restaurant_path = os.path.join(data_dir, restaurant)
    if os.path.isdir(restaurant_path):
        for img_file in os.listdir(restaurant_path):
            img_path = os.path.join(restaurant_path, img_file)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                doc = RestaurantImageDoc(
                    uri=img_path,
                    embedding=image_features[0].cpu(),
                    restaurant_name=restaurant
                )
                docs.append(doc)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Save DocList to disk
docs.save_binary("restaurant_index.bin")