# Streamlit Image Search with Open Flamingo and FAISS

## Overview

This project implements a **Streamlit-based image search application** that allows users to **search for images using text or voice input**. The system utilizes **Open Flamingo**, a multimodal AI model, and **FAISS** (Facebook AI Similarity Search) for ultra-fast retrieval of image descriptions.

The app:
- Supports **text and voice input** for searching images.
- Uses **Open Flamingo** to generate image descriptions.
- Uses **FAISS for fast similarity search**.
- Matches user queries to images using **natural language processing (NLP)**.
- **Plays back results using text-to-speech (TTS)**.
- Processes **batch image embeddings for speed**.
- Runs **locally** with a pre-downloaded model for fast performance.

---

## Features

- **🔎 Ultra-Fast Image Retrieval**: FAISS-powered similarity search.
- **🎙️ Voice Input Support**: Use voice commands to search.
- **🔊 Text-to-Speech**: Reads results aloud if enabled.
- **⚡ High-Speed Processing**: Batch processing speeds up image search.
- **📍 Local Execution**: No need for API calls; everything runs locally.
- **📂 Cached Image Descriptions**: Precompute and store results for quick lookup.
- **🚀 Multi-GPU Support**: Faster processing if multiple GPUs are available.

---

## Installation and Setup Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-repo/streamlit-image-search.git
cd streamlit-image-search
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Ensure Model is Downloaded
Place the **Open Flamingo model checkpoint** in the correct directory:
```bash
./OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt
```

### 4️⃣ Generate Image Descriptions
Before running FAISS, you must generate image descriptions:
```bash
python -c "from utils import generate_image_descriptions; generate_image_descriptions()"
```
Expected output:
```
🔄 Generating image descriptions...
✅ Cached X image descriptions.
```

### 5️⃣ Generate FAISS Index
After generating image descriptions, run:
```bash
python -c "from utils import refresh_faiss_index; refresh_faiss_index()"
```
Expected output:
```
🔄 Generating FAISS index...
✅ FAISS index successfully created at ./faiss_index.bin
```

### 6️⃣ Verify FAISS Index Exists
Check if FAISS has been saved:
```bash
ls -lah ./faiss_index.bin
```
- If the file **exists and is larger than 1MB**, FAISS is ready.
- If the file **does not exist**, re-run `refresh_faiss_index()`.

### 7️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## Project Structure

```
streamlit-image-search/
│── app.py             # Main Streamlit app
│── utils.py           # Helper functions for processing
│── config.py          # Configuration file for paths
│── requirements.txt   # Python dependencies
│── test_data_v2/      # Image dataset folder
│── OpenFlamingo-3B-vitl-mpt1b/  # Model checkpoint directory
│── faiss_index.bin    # FAISS index for fast search
│── image_descriptions.pkl  # Cached image descriptions
```

---

## Usage

### 🎤 Voice Search
1. Open the app (`streamlit run app.py`).
2. Select **Voice** as the input method.
3. Click **Record Voice** and speak your query (e.g., "A dog running in a park").
4. The app will retrieve and display the best matching image.
5. The result will be read aloud using TTS if enabled.

### ⌨️ Text Search
1. Open the app (`streamlit run app.py`).
2. Select **Text** as the input method.
3. Type your query (e.g., "A cat sleeping").
4. Click **Submit**.
5. The app will retrieve and display the best matching image.
6. The result will be read aloud using TTS if enabled.

### 🔄 Refresh FAISS Index
If new images are added, **refresh** FAISS and cache by clicking **Refresh Index** in the app or running:
```bash
python -c "from utils import refresh_faiss_index; refresh_faiss_index()"
```

---

## Technical Details

### 🔹 How It Works
1. **Preprocessing**: Images are processed in **batches** to generate descriptions using Open Flamingo.
2. **FAISS Indexing**: Generated image embeddings are stored in **FAISS** for fast retrieval.
3. **Search Mechanism**:
   - The system **converts user input into an embedding**.
   - The **best-matching image is retrieved** using FAISS similarity search.
4. **Voice Processing**:
   - Speech is converted to text using **Google Speech Recognition**.
   - The result is processed like a normal text query.
5. **Text-to-Speech**:
   - The app **reads the result aloud** using `gTTS` (Google Text-to-Speech) if enabled.

---

## Requirements

This app requires **Python 3.8+** and the following dependencies:

```
streamlit
faiss-cpu  # Use faiss-gpu for better performance if available
speechrecognition
torch==2.0.1
torchvision
transformers
huggingface_hub
numpy
matplotlib
gtts
Pillow
open_flamingo
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### ❓ FAISS Index Not Found
If FAISS is missing, regenerate it:
```bash
python -c "from utils import refresh_faiss_index; refresh_faiss_index()"
```

### ❓ No Images Found in Search
- Ensure `image_descriptions.pkl` exists.
- Run `generate_image_descriptions()` manually.
- Make sure your dataset is inside `test_data_v2/`.

### ❓ Voice Input Not Working
- Ensure **microphone access** is enabled.
- Try running:
  ```python
  import speech_recognition as sr
  sr.Microphone.list_microphone_names()
  ```
  and check if the correct microphone is detected.

### ❓ FAISS Index Not Refreshing
- Click **Refresh Index** in the Streamlit app.
- Run `generate_image_descriptions()` manually.

---

## Future Improvements

🚀 **Planned Features:**
- 🔥 **FAISS GPU integration** for ultra-fast search.
- 🌍 **Multilingual TTS support**.
- 📱 **Deploy as a mobile app** using Streamlit Cloud.

---

## Contributors

- **Your Name** - Dimakatso Mohapi
- **Community Contributions** - Always welcome!

---

## License

MIT License. Feel free to modify and use this project!

