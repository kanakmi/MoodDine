import streamlit as st
import torch
from transformers import AutoProcessor, AutoModel
from docarray import BaseDoc, DocList
from docarray.typing import TorchTensor
from docarray.utils.find import find

import base64
from PIL import Image
import io

def encode_image_to_base64(image_path):
    img = Image.open(image_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

st.set_page_config(
    page_title="MoodDine",
    page_icon="🍽️",
    layout="wide")

# Define the document schema
class RestaurantImageDoc(BaseDoc):
    uri: str
    embedding: TorchTensor
    restaurant_name: str

# Load processor and model
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
    model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Load saved image embeddings
@st.cache_resource
def load_index():
    return DocList[RestaurantImageDoc].load_binary("restaurant_index.bin")

search_docs = load_index()

# Streamlit UI
st.title("🍽️ MoodDine – match your dining mood with ambience")
query_text = st.text_input("What are you looking for?", placeholder="e.g., candlelight dinner, rooftop view, modern art decor")

if query_text:
    # Generate text embedding
    inputs = processor(text=query_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs, use_fast=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    query_embedding = text_features[0].cpu()

    # Search for top 10 matches
    matches, scores = find(
        index=search_docs,
        query=query_embedding,
        search_field="embedding",
        metric="cosine_sim",
        limit=10
    )

    # Show 3 distinct restaurants
    seen = set()
    shown = 0
    st.subheader("🛎️ Top Restaurant Matches")
    columns = st.columns(3)

    for match, score in zip(matches, scores):
        if match.restaurant_name not in seen:
            seen.add(match.restaurant_name)
            col_idx = shown % 3
            with columns[col_idx]:
                image_data = encode_image_to_base64(match.uri)
                card_html = f"""
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 15px;
                            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); text-align: center;">
                    <img src="{image_data}" style="width: 100%; border-radius: 12px;" />
                    <h4 style="margin-top: 10px;">Restaurant {shown+1}: {match.restaurant_name}</h4>
                    <p style="color: gray; font-size: 14px;">Similarity Score: {float(score):.4f}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                shown += 1
            if shown >= 3:
                break

    if shown == 0:
        st.warning("No distinct restaurants found. Try a different query.")
