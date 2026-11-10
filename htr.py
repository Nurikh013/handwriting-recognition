"""Word-level CRNN/BiLSTM/CTC training, evaluation and inference.

No downloads, training, package installation, or filesystem writes occur on import.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import string
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_ALPHABET = string.ascii_letters + string.digits + string.punctuation + " "
WORD_ID = re.compile(r"^[A-Za-z0-9]+-[A-Za-z0-9]+-[0-9]+-[0-9]+$")


@dataclass(frozen=True)
class Sample:
    word_id: str
    path: Path
    text: str
    group: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def minimum_ctc_steps(text: str) -> int:
    """Repeated adjacent labels need an intervening CTC blank."""
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def output_steps(width: int) -> int:
    if width < 8:
        raise ValueError("Image width must be at least 8 pixels.")
    return width // 4 - 1


def encode_text(text: str, alphabet: str) -> list[int]:
    mapping = {c: i + 1 for i, c in enumerate(alphabet)}
    unknown = set(text) - set(mapping)
    if unknown:
        raise ValueError(f"Characters outside the configured alphabet: {sorted(unknown)!r}")
    return [mapping[c] for c in text]


def collapse_ctc(indices: list[int], alphabet: str) -> str:
    out, previous = [], None
    for index in indices:
        if index < 0 or index > len(alphabet):
            raise ValueError(f"Invalid CTC class: {index}")
        if index and index != previous:
            out.append(alphabet[index - 1])
        previous = index
    return "".join(out)


def greedy_decode(logits: torch.Tensor, alphabet: str) -> list[str]:
    return [collapse_ctc(row, alphabet)
            for row in logits.argmax(2).transpose(0, 1).cpu().tolist()]


def preprocess_image(path: str | Path, width: int = 256) -> torch.Tensor:
    """Fit within 32 x width without cropping or horizontal distortion."""
    output_steps(width)
    with Image.open(path) as original:
        image = original.convert("L")
        scale = min(32 / image.height, width / image.width)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.BILINEAR)
        canvas = Image.new("L", (width, 32), 255)
        canvas.paste(image, (0, (32 - image.height) // 2))
        pixels = np.asarray(canvas, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(pixels).unsqueeze(0)


def load_samples(root: Path, alphabet: str, width: int, split_level: str = "form"):
    labels = root / "words.txt"
    if not labels.is_file():
        raise FileNotFoundError(f"Missing {labels}. See data/README.md; no dataset is downloaded automatically.")
    writers = {}
    if split_level == "writer":
        forms = root / "forms.txt"
        if not forms.is_file():
            raise FileNotFoundError("Writer splitting requires the official forms.txt (form ID, writer ID).")
        for line in forms.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and not line.startswith("#") and len(fields) >= 2:
                writers[fields[0]] = fields[1]
    samples, seen = [], set()
    rejected = dict(malformed=0, segmentation=0, missing=0, corrupt=0, empty=0, ctc=0)
    for line in labels.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=8)
        if len(parts) < 9 or not WORD_ID.fullmatch(parts[0]):
            rejected["malformed"] += 1
            continue
        word_id, status = parts[:2]
        if word_id in seen:
            raise ValueError(f"Duplicate annotation ID: {word_id}")
        seen.add(word_id)
        if status != "ok":
            rejected["segmentation"] += 1
            continue
        text = parts[8].replace("|", " ").strip()
        if not text:
            rejected["empty"] += 1
            continue
        encode_text(text, alphabet)  # Never silently drop unknown characters.
        if minimum_ctc_steps(text) > output_steps(width):
            rejected["ctc"] += 1
            continue
        a, b = word_id.split("-")[:2]
        form = a + "-" + b
        path = root / "words" / a / form / (word_id + ".png")
        if not path.is_file():
            rejected["missing"] += 1
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError):
            rejected["corrupt"] += 1
            continue
        if split_level == "writer" and form not in writers:
            raise ValueError(f"No writer metadata for form {form}")
        group = writers[form] if split_level == "writer" else form
        samples.append(Sample(word_id, path, text, group))
    if not samples:
        raise ValueError(f"No usable samples found. Rejected annotations: {rejected}")
    return samples, rejected


def split_samples(samples: list[Sample], seed: int = 42, max_samples: int = 0):
    """Split groups before subsampling; related crops never cross subsets."""
    groups = sorted({s.group for s in samples})
    if len(groups) < 3:
        raise ValueError("At least three independent groups are needed for train/validation/test.")
    rng = random.Random(seed)
    rng.shuffle(groups)
    train_n = min(max(1, int(len(groups) * 0.8)), len(groups) - 2)
    val_n = min(max(1, int(len(groups) * 0.1)), len(groups) - train_n - 1)
    names = {g: "train" if i < train_n else "val" if i < train_n + val_n else "test"
             for i, g in enumerate(groups)}
    subsets = {name: [] for name in ("train", "val", "test")}
    for sample in samples:
        subsets[names[sample.group]].append(sample)
    if max_samples:
        if max_samples < 3:
            raise ValueError("max_samples must be 0 (all samples) or at least 3.")
        # Allocate an exact global cap while retaining all three subsets.
        caps = {k: 1 for k in subsets}
        for _ in range(min(max_samples, len(samples)) - 3):
            options = [k for k in subsets if caps[k] < len(subsets[k])]
            k = max(options, key=lambda k: len(subsets[k]) / (caps[k] + 1))
            caps[k] += 1
        for k, values in subsets.items():
            rng.shuffle(values)
            subsets[k] = values[:caps[k]]
    return subsets


class HTRDataset(Dataset):
    def __init__(self, samples, alphabet, width):
        self.samples, self.alphabet, self.width = samples, alphabet, width

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        labels = torch.tensor(encode_text(sample.text, self.alphabet), dtype=torch.long)
        return preprocess_image(sample.path, self.width), labels, sample.text, sample.word_id


def collate_fn(batch):
    images, labels, texts, ids = zip(*batch)
    return (torch.stack(images), torch.cat(labels),
            torch.tensor([len(x) for x in labels], dtype=torch.long), list(texts), list(ids))


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU())
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or images.shape[2] != 32:
            raise ValueError("Expected image tensor (batch, 1, 32, width).")
        features = self.cnn(images).squeeze(2).permute(2, 0, 1)
        return self.fc(self.rnn(features)[0])


def edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        new = [i]
        for j, h in enumerate(hypothesis, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (r != h)))
        row = new
    return row[-1]


def error_rates(truths, predictions):
    if not truths or len(truths) != len(predictions) or any(not t for t in truths):
        raise ValueError("Metrics require matching, nonempty references and predictions.")
    chars = sum(len(t) for t in truths)
    words = sum(len(t.split()) for t in truths)
    return {"cer": sum(edit_distance(t, p) for t, p in zip(truths, predictions)) / chars,
            "wer": sum(edit_distance(t.split(), p.split()) for t, p in zip(truths, predictions)) / words,
            "exact_accuracy": sum(t == p for t, p in zip(truths, predictions)) / len(truths),
            "samples": len(truths)}


def ctc_loss(model, batch, device, criterion):
    images, labels, lengths, _, _ = batch
    logits = model(images.to(device))
    input_lengths = torch.full((len(images),), logits.size(0), dtype=torch.long)
    loss = criterion(logits.log_softmax(2), labels.to(device), input_lengths, lengths)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite CTC loss; check label lengths, inputs and model parameters.")
    return loss, logits


@torch.no_grad()
def evaluate(model, loader, alphabet, device):
    model.eval()
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    predictions, truths, records, loss_sum = [], [], [], 0.0
    for batch in loader:
        loss, logits = ctc_loss(model, batch, device, criterion)
        decoded = greedy_decode(logits, alphabet)
        predictions.extend(decoded); truths.extend(batch[3])
        records.extend({"id": i, "truth": t, "prediction": p}
                       for i, t, p in zip(batch[4], batch[3], decoded))
        loss_sum += loss.item() * len(decoded)
    metrics = error_rates(truths, predictions)
    metrics["loss"] = loss_sum / len(truths)
    return metrics, records


def save_checkpoint(path, model, alphabet, width, hidden, metadata=None):
    torch.save({"format_version": 1, "model_state": model.state_dict(),
                "alphabet": alphabet, "width": width, "hidden": hidden,
                "metadata": metadata or {}}, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format; use a checkpoint from this project version.")
    model = CRNN(len(checkpoint["alphabet"]) + 1, checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def loader(samples, alphabet, width, batch_size, shuffle=False, seed=42, workers=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(HTRDataset(samples, alphabet, width), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers, collate_fn=collate_fn,
                      generator=generator)


def run_training(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        raise ValueError("Epochs, batch size and learning rate must be positive; workers nonnegative.")
    alphabet = Path(args.alphabet_file).read_text(encoding="utf8").rstrip("\r\n") if args.alphabet_file else DEFAULT_ALPHABET
    if not alphabet or len(set(alphabet)) != len(alphabet):
        raise ValueError("The configured alphabet must be nonempty and have unique characters.")
    samples, rejected = load_samples(args.data_root, alphabet, args.width, args.split_level)
    splits = split_samples(samples, args.seed, args.max_samples)
    out = args.output
    if (out / "best.pt").exists():
        raise ValueError("This output folder already contains best.pt. Choose a new --output directory.")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {k: [{"id": s.word_id, "text": s.text, "group": s.group,
                     "path": s.path.relative_to(args.data_root).as_posix()} for s in v]
                for k, v in splits.items()}
    labels_hash = hashlib.sha256((args.data_root / "words.txt").read_bytes()).hexdigest()
    config = {"seed": args.seed, "width": args.width, "hidden": args.hidden,
              "split_level": args.split_level, "max_samples": args.max_samples,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "labels_sha256": labels_hash, "rejected": rejected,
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    (out / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    loaders = {k: loader(v, alphabet, args.width, args.batch_size, k == "train", args.seed, args.workers)
               for k, v in splits.items()}
    model = CRNN(len(alphabet) + 1, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = ctc_loss(model, batch, args.device, criterion)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += loss.item() * len(batch[0]); n += len(batch[0])
        metrics, _ = evaluate(model, loaders["val"], alphabet, args.device)
        scheduler.step(metrics["loss"])
        record = {"epoch": epoch, "train_loss": total / n, **{"val_" + k: v for k, v in metrics.items()}}
        history.append(record)
        print(json.dumps(record), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
        if metrics["cer"] < best:
            best = metrics["cer"]
            save_checkpoint(out / "best.pt", model, alphabet, args.width, args.hidden, config)
    model, _ = load_checkpoint(out / "best.pt", args.device)
    metrics, predictions = evaluate(model, loaders["test"], alphabet, args.device)
    (out / "test-metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    (out / "test-predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf8")
    print("Held-out test:", json.dumps(metrics))


def run_smoke(output: Path):
    """One synthetic training step; verifies plumbing, never handwriting accuracy."""
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    with tempfile.TemporaryDirectory() as temp:
        samples = []
        for i, word in enumerate(["a", "book"]):
            path = Path(temp) / f"{i}.png"
            image = Image.new("L", (64, 32), 255)
            ImageDraw.Draw(image).text((3, 8), word, fill=0)
            image.save(path)
            samples.append(Sample(str(i), path, word, str(i)))
        alphabet, width, hidden = "abok", 64, 16
        model = CRNN(len(alphabet) + 1, hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = next(iter(loader(samples, alphabet, width, 2)))
        loss, logits = ctc_loss(model, batch, "cpu", nn.CTCLoss())
        loss.backward(); optimizer.step()
        assert logits.shape == (output_steps(width), 2, len(alphabet) + 1)
        model.eval()
        with torch.no_grad(): before = model(batch[0])
        save_checkpoint(output / "smoke.pt", model, alphabet, width, hidden)
        restored, _ = load_checkpoint(output / "smoke.pt", "cpu")
        with torch.no_grad(): torch.testing.assert_close(before, restored(batch[0]))
        metrics, records = evaluate(restored, loader(samples, alphabet, width, 2), alphabet, "cpu")
        result = {"status": "passed", "training_steps": 1, "synthetic_samples": 2,
                  "loss_finite": True, "checkpoint_round_trip": True,
                  "logits_shape": list(logits.shape), "benchmark_accuracy_verified": False,
                  "torch_version": str(torch.__version__)}
        (output / "smoke-result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train with locally obtained IAM word data.")
    train.add_argument("--data-root", type=Path, default=Path("data/iam"))
    train.add_argument("--output", type=Path, default=Path("runs/experiment"))
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--width", type=int, default=256)
    train.add_argument("--hidden", type=int, default=256)
    train.add_argument("--max-samples", type=int, default=2000)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--split-level", choices=["form", "writer"], default="form")
    train.add_argument("--alphabet-file", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    predict = sub.add_parser("predict", help="Recognize one isolated word image.")
    predict.add_argument("image", type=Path)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--device", default="cpu")
    evaluate_cmd = sub.add_parser("evaluate", help="Evaluate a saved test split without re-splitting.")
    evaluate_cmd.add_argument("--data-root", type=Path, required=True)
    evaluate_cmd.add_argument("--checkpoint", type=Path, required=True)
    evaluate_cmd.add_argument("--split-file", type=Path, required=True)
    evaluate_cmd.add_argument("--device", default="cpu")
    smoke = sub.add_parser("smoke-test", help="One CPU training step on two synthetic images.")
    smoke.add_argument("--output", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args(argv)
    set_seed(getattr(args, "seed", 42))
    try:
        if args.command == "train":
            if args.hidden < 1 or args.max_samples < 0:
                raise ValueError("hidden must be positive; max_samples must be nonnegative.")
            run_training(args)
        elif args.command == "smoke-test":
            run_smoke(args.output)
        else:
            model, checkpoint = load_checkpoint(args.checkpoint, args.device)
            if args.command == "predict":
                with torch.no_grad():
                    logits = model(preprocess_image(args.image, checkpoint["width"]).unsqueeze(0).to(args.device))
                print(greedy_decode(logits, checkpoint["alphabet"])[0])
            else:
                records = json.loads(args.split_file.read_text(encoding="utf8"))["test"]
                root = args.data_root.resolve()
                samples = []
                for record in records:
                    path = (root / record["path"]).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError("Split path escapes the dataset directory.")
                    encode_text(record["text"], checkpoint["alphabet"])
                    if minimum_ctc_steps(record["text"]) > output_steps(checkpoint["width"]):
                        raise ValueError("An evaluation label cannot align to this checkpoint's image width.")
                    samples.append(Sample(record["id"], path, record["text"], record["group"]))
                metrics, _ = evaluate(model, loader(samples, checkpoint["alphabet"], checkpoint["width"], 32),
                                      checkpoint["alphabet"], args.device)
                print(json.dumps(metrics, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
