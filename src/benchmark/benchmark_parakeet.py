import sys
import json
import logging
import time
from pathlib import Path
import soundfile as sf

import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader

import nemo.collections.asr as nemo_asr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from runtime_repo.metric import score

DATA_FILE = PROJECT_ROOT / "data_files" / "validation_samples.csv"

LOG_FILE = "benchmark.log"
PREDICTION_FILE = "benchmark_predictions_parakeet.json"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

dots = "." * 15

print(f"Loading the Model{dots}")

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

MODEL = nemo_asr.models.ASRModel.from_pretrained(
    model_name=MODEL_NAME
)

MODEL = MODEL.to("cuda")
MODEL.eval()

class BenchmarkDataset(Dataset):
    def __init__(self, df, project_root, target_sr=16000):
        self.df = df.reset_index(drop=True).copy()
        self.df = self.df[self.df["orthographic_text"].str.strip() != ""].reset_index(drop=True)
        self.project_root = project_root
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]
        audio_path = str(self.project_root) + row["audio_path"]

        audio, sr = sf.read(audio_path)

        # Convert to torch tensor
        waveform = torch.tensor(audio)

        # If stereo → convert to mono
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)

        # Ensure shape is (time,)
        waveform = waveform.squeeze()

        return waveform, row["orthographic_text"]

def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    return waveforms, transcripts

def run_benchmark(loader, log_interval=500):

    predicted_list = []
    actual_list = []

    processed_samples = 0
    start_time = time.time()

    with torch.no_grad():

        for index, (waveforms, transcripts) in enumerate(loader):

            # Convert tensors to numpy for NeMo
            audio_data = [w.numpy().squeeze() for w in waveforms]

            predictions = MODEL.transcribe(
                audio=audio_data,
                batch_size=len(audio_data)
            )

            # Extract text from Hypothesis objects
            texts = []

            for p in predictions:
                if hasattr(p, "text"):
                    texts.append(p.text)
                elif isinstance(p, list):
                    texts.append(p[0].text)
                else:
                    texts.append(str(p))

            # Collect batch results
            predicted_list.extend(texts)
            actual_list.extend(transcripts)

            processed_samples += len(texts)

            print(f"Finished batch {index}")

            filtered_actual = [" ".join(actual_list)]
            filtered_predicted = [" ".join(predicted_list)]

            if processed_samples % log_interval < len(texts):
                partial_wer = score.score_wer(
                    filtered_actual,
                    filtered_predicted
                )

                elapsed = time.time() - start_time

                logging.info(
                    f"Samples: {processed_samples} | "
                    f"Partial WER: {partial_wer:.4f} | "
                    f"Time: {elapsed:.2f}s"
                )

                print(
                    f"Samples: {processed_samples} | "
                    f"Partial WER: {partial_wer:.4f}"
                )

    # Final filtering before returning (important)
    final_actual = []
    final_predicted = []

    for ref, hyp in zip(actual_list, predicted_list):
        if isinstance(ref, str) and ref.strip() != "":
            final_actual.append(ref.strip())
            final_predicted.append(hyp)

    return final_actual, final_predicted

print(f"Loading the CSV file{dots}")

df = pd.read_csv(DATA_FILE)

dataset = BenchmarkDataset(df, PROJECT_ROOT)

BATCH_SIZE = 32

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn,
)

logging.info(f"Model: {MODEL_NAME}")
logging.info(f"Batch Size: {BATCH_SIZE}")
logging.info(f"Number of Samples: {len(dataset)}")

print(f"Evaluating the model on validation set{dots}")

actual_list, predicted_list = run_benchmark(loader)
actual_list = [" ".join(actual_list)]
predicted_list = [" ".join(predicted_list)]
print(f"Calculating Final WER{dots}")

wer_value = score.score_wer(actual_list, predicted_list)

logging.info(f"Final WER: {wer_value}")

print("Final WER:", wer_value)

# Save predictions
results = [
    {"actual": a, "predicted": p}
    for a, p in zip(actual_list, predicted_list)
]

print(f"Saving predictions{dots}")

with open(PREDICTION_FILE, "w") as f:
    json.dump(results, f, indent=4)

logging.info(f"Predictions saved to {PREDICTION_FILE}")

print("Benchmark Complete")