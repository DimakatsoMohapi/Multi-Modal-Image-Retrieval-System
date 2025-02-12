import os
import pickle
import faiss
import torch
import numpy as np
import speech_recognition as sr
from PIL import Image
from gtts import gTTS
import tempfile
import streamlit as st
from torchvision import transforms
from config import DATASET_DIR, CACHE_FILE, FAISS_INDEX_PATH, model, image_processor, tokenizer, device

# 🛠️ Optimized Image Transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure consistent image size
    transforms.ToTensor()
])

# 🔎 Function to Find Best Match from FAISS
def find_best_match(user_text):
    print(f"🔎 Searching FAISS for: {user_text}")

    # Ensure FAISS index exists
    if not os.path.exists(FAISS_INDEX_PATH):
        print("❌ FAISS index file is missing!")
        return None

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    print("✅ FAISS index successfully loaded.")

    # Load cached image descriptions
    with open(CACHE_FILE, "rb") as cache_file:
        image_description_cache = pickle.load(cache_file)

    if len(image_description_cache) == 0:
        print("❌ No descriptions found in cache!")
        return None

    # Convert user text to an embedding
    lang_x = tokenizer([f"<image>This image is {user_text}"], return_tensors="pt").to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(lang_x["input_ids"], lang_x["attention_mask"]).cpu().numpy()

    print(f"🧠 Searching FAISS with text embedding shape: {text_embedding.shape}")

    # Perform FAISS search
    _, indices = index.search(np.array(text_embedding), 1)
    best_match_index = indices[0][0] if indices.size > 0 else None

    print(f"✅ FAISS Match Index: {best_match_index}")

    return image_description_cache[best_match_index][0] if best_match_index is not None else None


# 🖼️ Function to Generate Image Descriptions and Save Cache
def generate_image_descriptions(batch_size=32):  # Increased batch size for speed
    print("🔄 Generating image descriptions...")

    image_description_cache = []
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(DATASET_DIR)
        for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        print("❌ No images found in dataset!")
        return

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Fast image processing with torchvision
        vision_x = torch.stack([image_transform(Image.open(img).convert("RGB")) for img in batch_paths]).to(device)

        # Ensure correct shape for Open Flamingo
        vision_x = vision_x.half() if device == "cuda" else vision_x  # Convert to FP16 for speed
        vision_x = vision_x.unsqueeze(1).unsqueeze(2)  # Adds required dimensions (T_img=1, F=1)

        lang_x = tokenizer(["<image>This image is"] * len(batch_paths), return_tensors="pt", padding=True, truncation=True).to(device)

        print(f"📏 Processing batch {i // batch_size + 1}/{len(image_paths) // batch_size}")

        with torch.no_grad():
            try:
                generated_texts = model.generate(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                    max_new_tokens=30,  # Reduce for speed
                    num_beams=2,  # Reduce search complexity for faster generation
                )
                embeddings = model.get_image_features(vision_x).cpu().numpy()
            except Exception as e:
                print(f"❌ Error during model processing: {e}")
                return

        for img_path, gen_text, emb in zip(batch_paths, generated_texts, embeddings):
            description = tokenizer.decode(gen_text).strip()
            image_description_cache.append((img_path, description, emb))

    # Ensure directory exists before saving
    cache_dir = os.path.dirname(CACHE_FILE)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Save to CACHE_FILE
    with open(CACHE_FILE, "wb") as cache_file:
        pickle.dump(image_description_cache, cache_file)

    print(f"✅ Cached {len(image_description_cache)} image descriptions.")

# 🏎️ Function to Refresh FAISS Index
def refresh_faiss_index():
    print("🔄 Generating FAISS index...")

    if not os.path.exists(CACHE_FILE):
        print("❌ Cached image descriptions not found!")
        return

    # Load cached image descriptions
    with open(CACHE_FILE, "rb") as cache_file:
        image_description_cache = pickle.load(cache_file)

    # Ensure descriptions exist
    if len(image_description_cache) == 0:
        print("❌ No image descriptions found!")
        return

    # Initialize FAISS index
    index = faiss.IndexFlatL2(768)

    # Add embeddings to FAISS
    for _, _, emb in image_description_cache:
        index.add(np.array([emb]))

    # Ensure FAISS directory exists
    faiss_dir = os.path.dirname(FAISS_INDEX_PATH)
    if faiss_dir and not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Verify if FAISS was saved correctly
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"✅ FAISS index successfully created at {FAISS_INDEX_PATH}")
    else:
        print("❌ FAISS index failed to save!")

# 🔍 Function to Search and Display Results in Streamlit
def search_and_display(user_text):
    print(f"🔍 Searching for: {user_text}")  # Debugging statement
    matching_image = find_best_match(user_text)
    print(f"✅ Matching Image: {matching_image}")  # Debugging statement

    if matching_image:
        st.image(matching_image, caption=f"Match for '{user_text}'", use_column_width=True)
        if st.session_state.get("tts_enabled", False):
            tts_output = gTTS(user_text, lang='en')
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts_output.save(temp_audio.name)
            st.audio(temp_audio.name, format="audio/mp3")
    else:
        print("❌ No matching image found.")  # Debugging statement
        st.write("❌ No matching image found.")


# 🎤 Function to Recognize Speech for Voice Search
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening... Speak now.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"✅ Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        st.write("❌ Could not understand the audio.")
        return ""
    except sr.RequestError:
        st.write("❌ Could not request results. Check your internet connection.")
        return ""
