import sys
import json
import logging
import time
from pathlib import Path
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from runtime_repo.metric import score

DATA_FILE = PROJECT_ROOT / "data_files" / "validation_samples.csv"

LOG_FILE = "benchmark.log"
PREDICTION_FILE = "benchmark_predictions_whisper.json"


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
dots = "." * 15
print(f"Loading the Model{dots}")
MODEL_NAME = "openai/whisper-medium"

PROCESSOR = WhisperProcessor.from_pretrained(MODEL_NAME)
MODEL = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

MODEL.config.forced_decoder_ids = None
MODEL.to("cuda")
MODEL.eval()


class BenchmarkDataset(Dataset):
    def __init__(self, df, project_root, target_sr=16000):
        self.df = df.reset_index(drop=True).copy()
        self.project_root = project_root
        self.target_sr = target_sr
        self.resamplers = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio_path = f'{self.project_root}{row["audio_path"]}'

        audio, sr = sf.read(audio_path)
        waveform = torch.tensor(audio).unsqueeze(0)
        waveform = waveform.float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.target_sr
                )
            waveform = self.resamplers[sr](waveform)

        return waveform.squeeze(0), row["orthographic_text"]


def run_benchmark(loader, log_interval=500):

    predicted_list = []
    actual_list = []

    processed_samples = 0
    start_time = time.time()

    with torch.no_grad():

        for index, (waveforms, transcripts) in enumerate(loader):

            audio_data = [w.numpy().flatten() for w in waveforms]

            processed_batch = PROCESSOR(
                audio_data, sampling_rate=16000, return_tensors="pt"
            )

            input_features = processed_batch.input_features.to("cuda")

            predicted_ids = MODEL.generate(input_features)

            predictions = PROCESSOR.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            predicted_list.extend(predictions)
            actual_list.extend(transcripts)

            processed_samples += len(predictions)

            print(f"Finished batch {index}")

            # partial WER logging
            if processed_samples % log_interval < len(predictions):

                partial_wer = score.score_wer(actual_list, predicted_list)

                elapsed = time.time() - start_time

                logging.info(
                    f"Samples Processed: {processed_samples} | Partial WER: {partial_wer:.4f} | Time: {elapsed:.2f}s"
                )

                print(
                    f"Samples Processed: {processed_samples} | Partial WER: {partial_wer:.4f}"
                )

    return actual_list, predicted_list


def collate_fn(batch):
    waveforms = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    return waveforms, transcripts


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

print(f"Calculating Final WER{dots}")

wer_value = score.score_wer(actual_list, predicted_list)

logging.info(f"Final WER: {wer_value}")

print("Final WER:", wer_value)


results = [{"actual": a, "predicted": p} for a, p in zip(actual_list, predicted_list)]

print(f"Saving predictions{dots}")

with open(PREDICTION_FILE, "w") as f:
    json.dump(results, f, indent=4)

logging.info(f"Predictions saved to {PREDICTION_FILE}")

print("Benchmark Complete")
