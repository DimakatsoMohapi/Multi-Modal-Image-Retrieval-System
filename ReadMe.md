# Streamlit Image Search with Open Flamingo

## Overview

This project implements a **Streamlit-based image search application** that allows users to **search for images using text or voice input**. The system utilizes **Open Flamingo**, a multimodal AI model, to generate and retrieve image descriptions efficiently.

The app:

- Supports **text and voice input** for searching images.
- Uses **Open Flamingo** to generate image descriptions.
- Matches user queries to images using **natural language processing (NLP)**.
- **Plays back results using text-to-speech (TTS)**.
- Processes **batch image embeddings for speed**.
- Runs **locally** with a pre-downloaded model for fast performance.

---

## Features

- **🔎 Image Retrieval**: Find images based on user queries.
- **🎙️ Voice Input Support**: Use voice commands to search.
- **🔊 Text-to-Speech**: Reads results aloud.
- **⚡ Fast Processing**: Batch processing speeds up image search.
- **📍 Local Execution**: No need for API calls; everything runs locally.

---

## Installation

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

### 4️⃣ Run the Streamlit App

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
│── image_descriptions.pkl  # Cached image descriptions
```

---

## Usage

### 🎤 **Voice Search**

1. Open the app (`streamlit run app.py`).
2. Select **Voice** as the input method.
3. Click **Record Voice** and speak your query (e.g., "A dog running in a park").
4. The app will retrieve and display the best matching image.
5. The result will be read aloud using TTS.

### ⌨️ **Text Search**

1. Open the app (`streamlit run app.py`).
2. Select **Text** as the input method.
3. Type your query (e.g., "A cat sleeping").
4. The app will retrieve and display the best matching image.
5. The result will be read aloud using TTS.

---

## Technical Details

### 🔹 **How It Works**

1. **Preprocessing**: Images are processed in **batches** to generate descriptions using Open Flamingo.
2. **Caching**: Descriptions are **stored** in `image_descriptions.pkl` to speed up search.
3. **Search Mechanism**:
   - The system **compares user input** with generated descriptions.
   - The **best-matching image is retrieved** and displayed.
4. **Voice Processing**:
   - Speech is converted to text using **Google Speech Recognition**.
   - The result is processed like a normal text query.
5. **Text-to-Speech**:
   - The app **reads the result aloud** using `gTTS` (Google Text-to-Speech).

---

## Requirements

This app requires **Python 3.8+** and the following dependencies:

```
streamlit
speechrecognition
torch
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

### ❓ **App is Slow**

- Ensure you are using **GPU** (`torch.cuda.is_available()` should return `True`).
- Reduce `batch_size` in `generate_image_descriptions()`.
- Increase **num\_beams** in `model.generate()` for better performance.

### ❓ **No Images Found**

- Ensure `image_descriptions.pkl` exists.
- Run `generate_image_descriptions()` in `utils.py`.
- Make sure your dataset is inside `test_data_v2/`.

### ❓ **Voice Input Not Working**

- Ensure **microphone access** is enabled.
- Try running:
  ```python
  import speech_recognition as sr
  sr.Microphone.list_microphone_names()
  ```
  and check if the correct microphone is detected.

---

## Future Improvements

🚀 **Planned Features:**

- 🔥 **FAISS integration** for ultra-fast search.
- 🌍 **Multilingual TTS support**.
- 📱 **Deploy as a mobile app** using Streamlit Cloud.

---

## Contributors

- **Your Name** - Dimakatso Mohapi
- **Community Contributions** - Always welcome!

---

## License

MIT License. Feel free to modify and use this project!

