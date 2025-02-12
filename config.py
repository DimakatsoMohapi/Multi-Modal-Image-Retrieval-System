import torch
import os
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download

# Dataset Paths
DATASET_DIR = "./test_data_v2"
CACHE_FILE = "./image_descriptions.pkl"
FAISS_INDEX_PATH = "./faiss_index.bin"

# Device Selection (Force GPU for Speed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Open Flamingo Model (Proper Initialization)
def fast_model_init():
    print("Initializing Open Flamingo model...")
    
    # Create model and processors
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
    )

    # Load pre-downloaded model checkpoint
    checkpoint_path = "./OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"

    # Load checkpoint properly
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device).eval()

    print("✅ Model successfully initialized!")
    return model, image_processor, tokenizer

# Initialize Model
model, image_processor, tokenizer = fast_model_init()
