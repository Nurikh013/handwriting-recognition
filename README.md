# Handwritten Word Recognition

CRNN baseline for isolated English handwritten words: a convolutional encoder,
two-layer bidirectional LSTM, and CTC training/greedy decoding. The default
model follows the architecture in the accompanying project documents.

Project presentation name: **Irisbaev Nuriddin**.

**Validation status:** repaired code with automated tests and a synthetic CPU
training/checkpoint/inference check. IAM benchmark performance has not been
reproduced. The PDFs contain historical project results, explicitly marked as
unverified. No trained handwriting model or dataset is bundled.

## Install

Use Python 3.10-3.12 in a virtual environment. For a CPU installation:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA, follow the [official PyTorch installer](https://pytorch.org/get-started/locally/)
for your platform before installing the remaining requirements. The checked
local environment is recorded under `validation/`; the default commands use
CPU. GPU execution and numerical equivalence across devices were not tested.

## Verify without downloading IAM

```sh
python -m unittest discover -s tests -v
python htr.py smoke-test --output runs/smoke
python htr.py --help
```

The smoke test makes two tiny synthetic images, performs one training update,
checks tensor shapes and finite loss, saves/reloads a checkpoint, and evaluates
inference. It verifies software plumbing, not handwriting recognition accuracy.
Importing `htr` does not install packages, download data, train, or write files.

## Prepare the real dataset

Obtain IAM from its [official provider](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
and follow its registration and use terms. See [data/README.md](data/README.md)
for the expected directory structure and dataset citation. Data and Kaggle
credentials are excluded from version control. There is no automatic mirror
download or notebook-specific Google Drive mounting.

## Train

Quick experiment (up to 2,000 samples, eight epochs):

```sh
python htr.py train --data-root data/iam --output runs/quick
```

All available samples, 40 epochs, with writer grouping when `forms.txt` is present:

```sh
python htr.py train --data-root data/iam --output runs/full --max-samples 0 --epochs 40 --split-level writer
```

Add `--device cuda` only with a compatible installation. The default hidden
size is 256 per direction, image height is 32, width is 256, batch size is 32,
Adam learning rate is 0.001, and CTC blank is 0. Seed is 42. Worker count defaults
to 0 for Windows compatibility. Choose a new output folder for each run; an
existing `best.pt` will not be overwritten by training.

The default split groups by IAM form before using approximately 80%/10%/10%
of groups for train/validation/test. It is form-disjoint, not necessarily
writer-disjoint. `--split-level writer` groups by the writer IDs in `forms.txt`.
Group fractions do not imply exact sample fractions. These custom splits are
not an official IAM benchmark protocol. The saved `splits.json` records exact
IDs and groups; `config.json` records the annotation checksum and settings.
The maximum-sample cap is applied after splitting without mixing groups.

The default alphabet is fixed printable ASCII (letters, digits, punctuation,
space), defined before seeing any labels. For other characters supply a UTF-8
`--alphabet-file` with a single unique-character string. Unknown characters
raise an error instead of disappearing from the ground truth. Labels that
cannot fit CTC, including required blanks between repeated characters, are
filtered and counted. Missing, corrupt and invalid-segmentation images are
also counted. Very wide images are fitted proportionally and padded rather
than horizontally squeezed.

## Evaluate or predict

```sh
python htr.py evaluate --data-root data/iam --checkpoint runs/quick/best.pt --split-file runs/quick/splits.json
python htr.py predict path/to/word.png --checkpoint runs/quick/best.pt
```

Evaluation uses the saved test IDs instead of creating another random split.
Checkpoints include model weights, alphabet, image width and hidden size and
are loaded with `weights_only=True`. They are inference checkpoints, not
optimizer/resume checkpoints. Use checkpoints from this revised implementation;
the original notebook's checkpoint format is not assumed compatible.

Training exports `history.json`, `test-metrics.json` and `test-predictions.json`.
The best checkpoint is selected on validation CER. The test split is evaluated
after selection. CER/WER are corpus edit-distance ratios; exact accuracy uses
full-string equality. Empty predictions count as deletions, and insertions can
make error rates exceed 1. Greedy decoding produces text without an external
language model. No calibrated confidence or automatic acceptance threshold is
claimed by the revised CLI.

## Project documents and limitations

- [Presentation](docs/presentation.pdf): 12 historical slides, with updated
  project-name references and an unreproduced-results notice.
- [Project paper](docs/project-paper.pdf): five historical pages, with the
  same name update and a notice on every page. Research references are retained.
- [Results notes](docs/RESULTS.md): reported scores and missing evidence.
- [Validation](VALIDATION.md): repair details, completed checks and remaining limits.

The system expects a single cropped word, not a full page or handwritten line.
No full IAM training, accuracy reproduction, cross-writer benchmark, or GPU
performance claim is made by the preparation checks. To establish new results,
run the intended protocol, retain split manifests/checkpoints/logs, and update
the report from those measured outputs.

GitHub Actions runs the unit tests and synthetic integration checks on pushes
and pull requests. See the repository Actions tab for current results. Generated runs, checkpoints and dataset files are ignored.
No open-source license is selected for the code or project documents.
