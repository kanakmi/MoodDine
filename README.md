## MoodDine 🍽️
Discover restaurants based on the ambience you're dreaming of.

## Overview
When we search for restaurants today, we're limited by generic manual tags like "rooftop dining" or "candlelight dinner."
MoodDine lets you describe the exact vibe you're looking for — like "fairy lights" or "luxury dining with chandeliers" — and instantly surfaces real restaurant images that match your description.

Instead of scrolling through hundreds of images, you get exactly what you imagined — in one click.

## Key Features
- 🔍 Neural Search — not a generative AI project, no LLMs involved.

- 🧠 Powered by CLIP (jinaai/jina-clip-v2) — maps both images and text to a shared embedding space.

- ⚡ Fast Retrieval — thanks to DocArray by Jina AI.

- 🎨 Lightweight UI — built with Streamlit.

## Project Structure
```
.
├── app.py                # Streamlit App
├── embed_image.py        # Embeds restaurant images into vectors
├── search_image.py       # Quick script to verify search before running app
├── requirements.txt      # Project dependencies
├── restaurant_images/    # Folder containing restaurant ambience images
└── restaurant_index.bin  # (Generated) Binary file storing image embeddings
```

## How to Run Locally
1. Install Dependencies
```pip install -r requirements.txt```
Make sure you have Python 3.9+ installed.

2. Prepare Image Embeddings
Run the following script to generate image embeddings:
```python embed_image.py```
This will create a restaurant_index.bin file in your project directory.

3. Test Neural Search (Optional but Recommended)
You can quickly test if the embeddings are correctly working:
```python search_image.py```
It will prompt you to enter a text query and print top matching restaurant names and scores.

4. Launch the Streamlit App
Finally, run the app:
```streamlit run app.py```
Open the Streamlit URL shown in the terminal — and start searching based on vibes!

## Notes
- No cloud services or GPUs are needed — runs comfortably on CPU as well.
- You can add your own restaurant images inside the restaurant_images/ folder to expand the database.
- Make sure your images are clear ambience shots (not food photos) for best results.

## Tech Stack
- Streamlit
- Jina AI - DocArray
- Hugging Face Transformers
- CLIP Model (jinaai/jina-clip-v2)

<b> Built with ❤️ over a weekend. </b>

