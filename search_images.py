import torch
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


# Load it back
search_docs = DocList[RestaurantImageDoc].load_binary("restaurant_index.bin")

# --- QUERY PHASE ---

query_text = "place with modern art paintings"
inputs = processor(text=query_text, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**inputs, use_fast=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

query_embedding = text_features[0].cpu()

# Perform vector search
matches, scores = find(
    index=search_docs,
    query=query_embedding,
    search_field="embedding",
    metric="cosine_sim",
    limit=5
)

# Show results
print(f"\nTop matches for query: '{query_text}'")
for match, score in zip(matches, scores):
    print(f"Restaurant: {match.restaurant_name}, Image: {match.uri}, Similarity Score: {float(score):.4f}")

