from __future__ import annotations

import os
from pathlib import Path

import gdown
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = "./model"

# Label mapping  (adjust if your model uses different ids)
ID2LABEL = {0: "Irrelevant", 1: "Relevant"}


def download_model_if_missing(drive_url: str, model_path: str = "./model"):
    """Download model.safetensors from Google Drive if not already present."""
    model_file = os.path.join(model_path, "model.safetensors")
    os.makedirs(model_path, exist_ok=True)
    if not os.path.exists(model_file):
        print("Downloading model from Google Drive...")
        gdown.download(drive_url, model_file, fuzzy=True)
        print("Download complete.")


class Predictor:
    """Loads the fine-tuned IndoBERT-Relevancy model and runs inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load tokenizer and model from local path.
        Returns True on success, False on failure.
        Call .error to get the error message.
        """
        if self._loaded:
            return True

        if not self.model_path.exists():
            self._error = (
                f"Model directory not found: `{self.model_path.resolve()}`.\n\n"
                "Please place the fine-tuned model files inside the `model/` folder:\n"
                "- `config.json`\n"
                "- `tokenizer.json` / `tokenizer_config.json`\n"
                "- `pytorch_model.bin` or `model.safetensors`"
            )
            return False

        required_files = ["config.json"]
        missing = [f for f in required_files if not (self.model_path / f).exists()]
        if missing:
            self._error = (
                f"Missing required model files: {', '.join(missing)}.\n\n"
                "Ensure all model artifacts are present in the `model/` directory."
            )
            return False

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            self._error = None
            return True
        except Exception as exc:
            self._error = (
                f"Failed to load model: {exc}\n\n"
                "Please verify the model files are valid and compatible with "
                "the installed `transformers` version."
            )
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, video_title: str, comment: str) -> dict:
        """
        Predict relevancy for a single (video_title, comment) pair.
        Input text format: "<video_title> [SEP] <comment>"
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        text = f"{video_title} [SEP] {comment}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

        if len(probs) == 2:
            label_0 = ID2LABEL.get(0, "Irrelevant")
            label_1 = ID2LABEL.get(1, "Relevant")
            scores = {label_0: probs[0], label_1: probs[1]}
            pred_idx = int(torch.argmax(torch.tensor(probs)).item())
            predicted_label = ID2LABEL.get(pred_idx, "Unknown")
            confidence = probs[pred_idx]
        else:
            prob = probs if isinstance(probs, float) else probs[0]
            predicted_label = "Relevant" if prob >= 0.5 else "Irrelevant"
            confidence = prob if predicted_label == "Relevant" else 1 - prob
            scores = {"Relevant": prob, "Irrelevant": 1 - prob}

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
        }

    def predict_batch(
        self,
        video_title: str,
        comments: list[str],
        batch_size: int = 16,
        progress_callback=None,
    ) -> list[dict]:
        """
        Run prediction for a list of comments in batches.
        progress_callback(current, total) is called after each batch.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        results = []
        total = len(comments)

        for i in range(0, total, batch_size):
            batch_comments = comments[i : i + batch_size]
            texts = [f"{video_title} [SEP] {c}" for c in batch_comments]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu()

            for j in range(probs.shape[0]):
                p = probs[j].tolist()
                pred_idx = int(torch.argmax(probs[j]).item())
                label_0 = ID2LABEL.get(0, "Irrelevant")
                label_1 = ID2LABEL.get(1, "Relevant")
                scores = {label_0: round(p[0], 4), label_1: round(p[1], 4)}
                predicted_label = ID2LABEL.get(pred_idx, "Unknown")
                confidence = round(p[pred_idx], 4)
                results.append(
                    {
                        "label": predicted_label,
                        "confidence": confidence,
                        "scores": scores,
                    }
                )

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return results
