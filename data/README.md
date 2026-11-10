# IAM data setup

Obtain the IAM Handwriting Database from its official provider:

https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

The official site requests registration and describes non-commercial research
use. Follow those terms; a third-party mirror does not replace them. No IAM
images, annotations, Kaggle credentials, or downloaded archives are included
in this prepared repository. The synthetic tests create their own images.

Expected layout:

```text
data/iam/
  words.txt
  forms.txt                 # needed only for --split-level writer
  words/
    a01/
      a01-000u/
        a01-000u-00-00.png
        ...
```

`words.txt` uses the official IAM word format: word ID, segmentation status,
gray level, bounding-box x/y/width/height, grammatical tag, transcription.
The parser reads the transcription after the first eight fields, skips `err`
entries, validates image paths and rejects duplicate IDs. `forms.txt` maps
the form ID in column 1 to the writer ID in column 2.

Default grouping uses the form part of each word ID. Writer grouping requires
complete form-to-writer metadata. Neither is advertised as an official IAM
partition. Record the exact data version and use the saved split manifests
when comparing experiments.

Dataset citation requested by the provider:

U.-V. Marti and H. Bunke. *The IAM-database: An English Sentence Database for
Off-line Handwriting Recognition*. International Journal on Document Analysis
and Recognition, 5, 39-46, 2002.
