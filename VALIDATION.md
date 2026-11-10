# Preparation and validation record

## Original archive findings

The source archive contained a presentation, a five-page project paper, and a
Python file inside another ZIP, plus macOS resource-fork files. It contained no
dataset, trained checkpoint, environment manifest, test suite, or training logs.

The Python file was not a valid standalone program: it began with a notebook
`!pip` command and the `decode_indices` definition was missing its colon. It
also used `/content/` paths, ran downloads and training at import time, lacked
a Windows entry-point guard, and used fragile symlinks for mirrored data.

Data/evaluation issues included unknown characters being silently discarded,
no check for the extra CTC steps required by adjacent repeats, missing-image
checks without corrupt-image checks, vocabulary construction before splitting,
random crop-level splitting, and image-width clamping that could distort wide
words. Empty predictions were replaced with spaces for scoring. The saved
checkpoint did not record enough preprocessing/model settings for an independent
prediction command.

## Prepared version

The nested code archive and macOS artifacts are replaced by a normal `htr.py`
entry point, a test suite, dependency instructions, dataset documentation and a
GitHub Actions configuration. The CRNN CNN/BiLSTM architecture is retained,
while training, input validation, metrics, splitting and checkpoint handling
are refactored into testable functions and explicit CLI commands.

The fixed alphabet is declared without reading evaluation labels; unknown
characters raise errors. Forms stay in one subset, and writer metadata can be
used instead. Split IDs, annotation checksum, settings and run metrics are
saved. CTC-infeasible labels are counted before training; infinite losses are
not silently zeroed. Preprocessing fits both image dimensions proportionally.
Importing the module has no download/training side effects. Checkpoints are
loaded with restricted tensor-oriented deserialization.

The presentation and paper list Irisbaev Nuriddin in the project-name positions;
other project-name references and the generic presentation-author metadata have
been removed. Research citations remain. All pages are retained, with explicit
notices that historical results were not reproduced.

## Verification

All 15 local tests passed on CPU using Python 3.12.14, PyTorch 2.6.0+cpu,
NumPy 2.3.5 and Pillow 13.0.0. Details are recorded in `validation/`.
The suite tests decoding, CTC alignment, unknown characters,
corpus metrics, grouped splits, sample caps, image normalization, annotation
filtering, duplicate IDs, writer-metadata requirements, model output shape,
one optimizer step, checkpoint reload equality and inference.

GitHub Actions runs the same test suite on pushes and pull requests. Full IAM
training, GPU execution, benchmark accuracy and confidence calibration are not
validated by the synthetic checks. See `docs/RESULTS.md` for the distinction
between the historical reported figures and the checked software behavior.

The original desktop ZIP is preserved separately and is not bundled in the
prepared upload copy. The files in this repository are the prepared project copy.
