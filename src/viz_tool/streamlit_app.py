import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from pathlib import Path 
import sys 

# path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Adjust these to your actual file names
base_model_preds = PROJECT_ROOT / "src" / "benchmark" / "benchmark_predictions_parakeet.json"
finetune_model_preds = PROJECT_ROOT / "src" / "benchmark" / "predictions_finetune_parakeet_won.json"

# Data loading
def score_sentences(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as file:
        return json.load(file)

# This block runs ONLY ONCE when the app starts.
if 'data' not in st.session_state:
    finetune_model_contents = score_sentences(str(finetune_model_preds))
    
    initial_data = []
    for index, item in enumerate(finetune_model_contents):
        initial_data.append({
            "id": index,
            "audio_path": item.get("audio_filepath", ""),
            "reference": item.get("reference", ""),
            "finetune_model_pred": item.get("prediction", ""),
            "comments": ""
        })
    
    st.session_state.data = initial_data
    st.session_state.index = 0

def update_data():
    """Syncs the text area content into the master data list."""
    idx = st.session_state.index
    st.session_state.data[idx]['comments'] = st.session_state[f"input_{idx}"]

def change_index(new_index):
    """Safely moves to a new sample index."""
    st.session_state.index = new_index

st.set_page_config(page_title="ASR Viz Tool", layout="wide")
st.title("🧪 ASR Viz Tool: Children's Speech Analysis")
st.markdown(f"Currently analyzing **{len(st.session_state.data)}** Samples.")

#siebar
with st.sidebar:
    st.header("Navigation & Export")
    
    # Number input tied to session state
    new_idx = st.number_input(
        "Go to Sample Index", 
        0, len(st.session_state.data)-1, 
        value=st.session_state.index
    )
    if new_idx != st.session_state.index:
        st.session_state.index = new_idx
        st.rerun()

    st.divider()
    
    # Export Button
    if st.button("💾 Export All Findings to JSON", use_container_width=True):
        # Final safety sync
        update_data()
        save_path = "asr_report.json"
        with open(save_path, "w") as f:
            json.dump(st.session_state.data, f, indent=4)
        st.success(f"Successfully saved to {save_path}!")

sample = st.session_state.data[st.session_state.index]

col1, col2 = st.columns([1, 1])

# Column 1: Audio & Spectrogram
with col1:
    st.subheader("1. Audio & Acoustic Features")
    audio_path = sample['audio_path']
    
    if os.path.exists(audio_path):
        st.audio(audio_path)
        
        # Plotting Mel-Spectrogram
        y, sr = librosa.load(audio_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f"Sample Index: {st.session_state.index}")
        st.pyplot(fig)
    else:
        st.error(f"File not found: {audio_path}")

# Column 2: Predictions & Comments
with col2:
    st.subheader("2. Textual Comparison")
    st.info(f"**Reference (Ground Truth):**\n\n {sample['reference']}")
    st.success(f"**Finetuned Model Prediction:**\n\n {sample['finetune_model_pred']}")
    
    st.divider()
    
    st.subheader("3.Comments")
    # THE KEY PART: 'value' pulls from state, 'on_change' saves to state
    st.text_area(
        "Add comments here.",
        value=sample['comments'],
        key=f"input_{st.session_state.index}",
        on_change=update_data,
        height=250,
    )

# Bottom Navigation
st.divider()
b1, b2, _ = st.columns([1, 1, 4])

if b1.button("⬅️ Previous") and st.session_state.index > 0:
    st.session_state.index -= 1
    st.rerun()

if b2.button("Next ➡️") and st.session_state.index < len(st.session_state.data) - 1:
    st.session_state.index += 1
    st.rerun()