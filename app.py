import streamlit as st
from utils import generate_image_descriptions, find_best_match, recognize_speech, search_and_display, refresh_faiss_index
from config import model, image_processor, tokenizer, device

# Streamlit UI
st.title("🔍 Ultra-Fast Image Search with FAISS & Open Flamingo")

# Sidebar with Refresh Index Option
if st.sidebar.button("🔄 Refresh FAISS Index"):
    refresh_faiss_index()
    st.sidebar.success("FAISS index refreshed successfully!")

# User Input Selection
input_method = st.radio("Choose Input Method:", ("Text", "Voice"))

user_text = ""
if input_method == "Text":
    user_text = st.text_input("Enter a description (e.g., 'a cat'):")
    if st.button("🔍 Submit"):
        search_and_display(user_text)

elif input_method == "Voice":
    if st.button("🎤 Record Voice"):
        user_text = recognize_speech()
        if user_text:
            search_and_display(user_text)

# Enable or Disable Text-to-Speech
tts_enabled = st.checkbox("🔊 Enable Text-to-Speech")
if tts_enabled:
    st.session_state.tts_enabled = True
else:
    st.session_state.tts_enabled = False
