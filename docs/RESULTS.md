# Historical results and reproduction status

The supplied paper and slides report these historical values:

| Quantity | Reported value | Evidence available in the original archive |
| --- | --- | --- |
| Validation CER | 0.012 | Paper/slide text and figure images |
| Validation WER | 0.027 | Paper/slide text and figure images |
| Exact-match test accuracy | 179/200 = 89.5% | Reported 200-word subset and figure |
| Training duration | 100 epochs | Paper table and historical curves |

These values have **not been reproduced or independently verified** during
repository preparation. No original weights, training logs, exact split IDs,
random-sample list, or input dataset were supplied. The validation scores and
the 200-word test accuracy concern different reported populations and should
not be presented as metrics on one identical test set.

The supplied code defaulted to a 2,000-sample demonstration with eight epochs
(40 epochs outside demonstration mode), while the paper describes 100 epochs.
The original code built its vocabulary before splitting and used random word
splits, so its implementation did not match a training-only-vocabulary or
writer-independent evaluation claim. It also lacked the confidence calculation
discussed in the paper. Those historical claims are not new measurements from
the revised code.

The revised implementation uses a predeclared alphabet, form- or writer-grouped
splits, validated CTC lengths, explicit preprocessing settings and recorded split
IDs. These corrections change the experiment, so its future scores must be
reported separately. There is no guarantee that the original scores will be
recovered after training.

Current checks cover unit behavior, tensor dimensions, finite CTC loss, one
synthetic optimizer step, checkpoint round-trip equality and inference. The
synthetic test's predictions and loss are not benchmark results. Its temporary
checkpoint is not a usable handwriting model and is not included in the package.

The PDFs retain their historical figures and research references. A notice in
each document identifies their results as unreproduced. Project-name references
have been changed to Irisbaev Nuriddin; this editorial update does not verify
the original experiment or establish new performance claims.
