import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --- CONFIG & STYLING ---
st.set_page_config(page_title="ASR Research Partner", layout="wide")
st.title("🧪 ASR Forensics: Children's Speech Analysis")
st.markdown("Compare **Nemoparakeet 0.6B** vs. **Finetuned Model** to bridge the 0.23 WER gap.")

# --- DATA INITIALIZATION ---
# Replace this 'dummy_data' with your actual list of dicts from your benchmarking script
if 'data' not in st.session_state:
    # Example format: 
    st.session_state.data = [
        {
            "id": "sample_001",
            "audio_path": "path/to/your/audio.wav", 
            "reference": "the quick brown fox jumps over the lazy dog",
            "model_1_bench": "the quick brown box jumps over the lady dog",
            "model_2_fine": "the quick brown fox jumps over the lazy dog",
            "analysis": ""
        }
        # ... add all your samples here
    ]
    st.session_state.index = 0

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Navigation & Export")
st.session_state.index = st.sidebar.number_input(
    "Go to Sample Index", 0, len(st.session_state.data)-1, st.session_state.index
)

if st.sidebar.button("💾 Export All Findings to JSON"):
    with open("asr_forensics_report.json", "w") as f:
        json.dump(st.session_state.data, f, indent=4)
    st.sidebar.success("Exported to asr_forensics_report.json!")

# --- MAIN UI ---
sample = st.session_state.data[st.session_state.index]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Audio & Acoustic Features")
    if os.path.exists(sample['audio_path']):
        st.audio(sample['audio_path'])
        
        # Plotting Mel-Spectrogram
        y, sr = librosa.load(sample['audio_path'])
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f"Mel Spectrogram - {sample['id']}")
        st.pyplot(fig)
    else:
        st.error(f"Audio file not found: {sample['audio_path']}")

with col2:
    st.subheader("2. Textual Comparison")
    st.markdown("---")
    st.info(f"**Reference (Ground Truth):**\n\n {sample['reference']}")
    st.error(f"**Model 1 (Benchmark 0.6B):**\n\n {sample['model_1_bench']}")
    st.success(f"**Model 2 (Finetuned):**\n\n {sample['model_2_fine']}")
    
    st.markdown("---")
    st.subheader("3. Researcher Comments")
    # Capturing input and saving it directly to session state
    user_comment = st.text_area(
        "Enter Analysis (e.g., 'Deletion due to low volume', 'Insertion in background noise')",
        value=sample['analysis'],
        key=f"comment_{st.session_state.index}",
        height=200
    )
    
    if st.button("Save Comment for this Sample"):
        st.session_state.data[st.session_state.index]['analysis'] = user_comment
        st.toast("Comment saved to memory!")

# Quick Navigation Buttons at bottom
b_col1, b_col2, _ = st.columns([1, 1, 4])
if b_col1.button("⬅️ Previous") and st.session_state.index > 0:
    st.session_state.index -= 1
    st.rerun()
if b_col2.button("Next ➡️") and st.session_state.index < len(st.session_state.data) - 1:
    st.session_state.index += 1
    st.rerun()