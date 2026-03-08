import json
import os
import sys
import logging
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torchaudio

os.environ["NUMBA_CUDA_DEFAULT_PTX_CC"] = "8.0"

from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from omegaconf import OmegaConf, open_dict
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.finetune.nemo_adapter import (
    add_global_adapter_cfg,
    patch_transcribe_lhotse,
    update_model_cfg,
    update_model_config_to_support_adapter,
)
from src.preprocessing.cleaning import clean_audio
from src.finetune.score import english_spelling_normalizer, score_wer
import soundfile as sf


DATA_FILE = PROJECT_ROOT / "data_files" / "validation_samples.csv"
MANIFEST_DIR = PROJECT_ROOT / "processed" / "ortho_dataset_val"

VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
PREDICTIONS_FILE = "predictions_fientune_parakeet.json"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "finetune_test.log"

torch.set_float32_matmul_precision("high")

def setup_logging():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


def prepare_data(sample=None):

    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    df["audio_path"] = df["audio_path"].apply(
        lambda path: f"{PROJECT_ROOT}{path}"
    )

    df = df[
        ["audio_path", "audio_duration_sec", "orthographic_text"]
    ].rename(
        columns={
            "audio_path": "audio_filepath",
            "audio_duration_sec": "duration",
            "orthographic_text": "text",
        }
    )

    logger.info(f"Loaded {len(df)} samples")

    if sample:
        logger.info(f"Sampling {sample} samples")
        df = df.sample(sample, random_state=0)

    df.to_json(VAL_MANIFEST, orient="records", lines=True)

    logger.info(f"Validation samples: {len(df)}")

    return VAL_MANIFEST


def build_config(val_manifest):

    logger.info("Building training configuration...")

    BATCH_SIZE = 4
    NUM_WORKERS = 4

    yaml_path = PROJECT_ROOT / "src" / "finetune" / "asr_adaptation.yaml"
    cfg = OmegaConf.load(yaml_path)

    overrides = OmegaConf.create(
        {
            "model": {
                "pretrained_model": "nvidia/parakeet-tdt-0.6b-v2",
                "adapter": {
                    "adapter_name": "asr_children_orthographic",
                    "adapter_module_name": "encoder",
                    "linear": {"in_features": 1024},
                },
                "validation_ds": {
                    "manifest_filepath": str(val_manifest),
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "use_lhotse": False,
                    "channel_selector": "average",
                },
                "optim": {
                    "lr": 0.001,
                    "weight_decay": 0.0,
                },
            }
        }
    )

    cfg = OmegaConf.merge(cfg, overrides)

    logger.info("Configuration ready")

    return cfg

def clean_audio_files(audio_files):

    cleaned_audio = []

    for index, path in enumerate(audio_files):
        audio, sr = sf.read(path)
        audio = torch.tensor(audio)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        audio = clean_audio(audio, sample_rate=16000)

        cleaned_audio.append(audio.cpu().numpy())
        if index % 500 == 0:
            print("Denoised : {} Files".format(index))
    return cleaned_audio

def evaluate_model(exp_log_dir, cfg):

    logger.info("Starting evaluation...")

    exp_log_dir = Path(exp_log_dir)

    nemo_ckpts = sorted((exp_log_dir / "checkpoints").glob("*.nemo"))

    if not nemo_ckpts:
        raise FileNotFoundError("No .nemo checkpoints found")

    best_ckpt = nemo_ckpts[-1]

    logger.info(f"Loading checkpoint: {best_ckpt}")

    eval_model = ASRModel.restore_from(best_ckpt, map_location="cuda")

    with open_dict(eval_model.cfg):
        eval_model.cfg.decoding.greedy.use_cuda_graph_decoder = False

    eval_model.change_decoding_strategy(eval_model.cfg.decoding)

    patch_transcribe_lhotse(eval_model)

    with open(cfg.model.validation_ds.manifest_filepath) as f:
        val_entries = [json.loads(line) for line in f]

    audio_files = [e["audio_filepath"] for e in val_entries]
    references = [e["text"] for e in val_entries]

    logger.info(f"Running inference on {len(audio_files)} files")
    cleaned_audio_files = clean_audio_files(audio_files)
    raw = eval_model.transcribe(
        cleaned_audio_files,
        batch_size=cfg.model.validation_ds.batch_size,
        channel_selector="average",
        verbose=False,
    )

    if isinstance(raw, tuple):
        raw = raw[0]

    predictions = [h.text if hasattr(h, "text") else h for h in raw]

    normalizer = EnglishTextNormalizer(english_spelling_normalizer)

    filtered = [
        (audio, r, p)
        for audio, r, p in zip(audio_files, references, predictions)
        if normalizer(r) != ""
    ]

    audio_files, references, predictions = zip(*filtered)

    wer = score_wer(references, predictions)

    logger.info(f"Validation WER: {wer:.4f}")

    logger.info("Sample predictions:")

    for ref, pred in zip(references[:5], predictions[:5]):
        logger.info(f"REF:  {ref}")
        logger.info(f"PRED: {pred}")

    logger.info("Saving predictions...")

    results = []
    for audio, ref, pred in zip(audio_files, references, predictions):
        results.append(
            {
                "audio_filepath": audio,
                "reference": ref,
                "prediction": pred,
            }
        )

    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Predictions saved to {PREDICTIONS_FILE}")


def main():

    logger.info("Evaluation Started....")

    val_manifest = prepare_data()

    cfg = build_config(val_manifest)

    exp_log_dir = "/home/chandan/ASRProject/models/orthographic_finetune_nemo/ASR-Adapter/2026-03-07_18-42-07"

    evaluate_model(exp_log_dir, cfg)

    logger.info("========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()