import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

import htr


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_ctc_collapse_repeats_and_blank(self):
        self.assertEqual(htr.collapse_ctc([1, 1, 0, 1, 2, 2], "ab"), "aab")
        self.assertEqual(htr.collapse_ctc([0, 0], "ab"), "")

    def test_ctc_alignment_accounts_for_repeated_characters(self):
        self.assertEqual(htr.minimum_ctc_steps("book"), 5)
        self.assertEqual(htr.minimum_ctc_steps("aaa"), 5)
        self.assertEqual(htr.output_steps(256), 63)

    def test_unknown_character_is_not_silently_deleted(self):
        with self.assertRaises(ValueError):
            htr.encode_text("az", "ab")

    def test_metrics_include_empty_predictions_and_insertions(self):
        metrics = htr.error_rates(["cat", "a"], ["", "abc"])
        self.assertEqual(metrics["cer"], 5 / 4)
        self.assertEqual(metrics["wer"], 1)
        self.assertEqual(metrics["exact_accuracy"], 0)
        self.assertEqual(htr.error_rates(["book"], ["book"])["cer"], 0)

    def test_metrics_reject_empty_reference_collection(self):
        with self.assertRaises(ValueError):
            htr.error_rates([], [])

    def test_split_has_disjoint_groups_and_reproducible_ids(self):
        samples = [htr.Sample(f"{group}-{i}", Path("unused"), "a", str(group))
                   for group in range(12) for i in range(4)]
        first = htr.split_samples(samples, seed=4)
        second = htr.split_samples(samples, seed=4)
        self.assertEqual(first, second)
        groups = [{s.group for s in first[k]} for k in ("train", "val", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(map(len, first.values())), len(samples))

    def test_subsampling_respects_global_cap_and_nonempty_subsets(self):
        samples = [htr.Sample(f"{group}-{i}", Path("unused"), "a", str(group))
                   for group in range(10) for i in range(6)]
        split = htr.split_samples(samples, max_samples=8)
        self.assertEqual(sum(map(len, split.values())), 8)
        self.assertTrue(all(split.values()))

    def test_insufficient_groups_are_reported(self):
        with self.assertRaises(ValueError):
            htr.split_samples([htr.Sample("id", Path("x"), "a", "one")])

    def test_preprocessing_preserves_aspect_and_white_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wide.png"
            Image.new("L", (800, 20), 0).save(path)
            tensor = htr.preprocess_image(path, 64)
            self.assertEqual(tuple(tensor.shape), (1, 32, 64))
            self.assertEqual(tensor[0, 0, 0].item(), 1)  # top padding, no squeezing
            self.assertEqual(tensor[0, 15, 0].item(), -1)
            self.assertTrue(torch.isfinite(tensor).all())

    def test_loader_filters_segmentation_missing_corrupt_and_unalignable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "words/a01/a01-000u"
            folder.mkdir(parents=True)
            Image.new("L", (40, 20), 255).save(folder / "a01-000u-00-00.png")
            (folder / "a01-000u-00-03.png").write_bytes(b"not a png")
            (root / "words.txt").write_text(
                "# header\n"
                "a01-000u-00-00 ok 0 0 0 0 0 NN book\n"
                "a01-000u-00-01 err 0 0 0 0 0 NN a\n"
                "a01-000u-00-02 ok 0 0 0 0 0 NN a\n"
                "a01-000u-00-03 ok 0 0 0 0 0 NN a\n"
                "a01-000u-00-04 ok 0 0 0 0 0 NN aaaaaaaaaa\n"
                "../../escape ok 0 0 0 0 0 NN a\n", encoding="utf8")
            samples, rejected = htr.load_samples(root, "abok", 32)
            self.assertEqual([s.text for s in samples], ["book"])
            for key in ("segmentation", "missing", "corrupt", "ctc", "malformed"):
                self.assertEqual(rejected[key], 1, key)

    def test_duplicate_annotations_are_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            line = "a01-000u-00-00 ok 0 0 0 0 0 NN a\n"
            (root / "words.txt").write_text(line * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                htr.load_samples(root, "a", 64)

    def test_writer_split_requires_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "words.txt").write_text("")
            with self.assertRaisesRegex(FileNotFoundError, "forms.txt"):
                htr.load_samples(root, "a", 64, "writer")

    def test_model_time_axis_matches_preprocessor_width(self):
        model = htr.CRNN(5, hidden=16).eval()
        with torch.no_grad(): logits = model(torch.zeros(2, 1, 32, 64))
        self.assertEqual(tuple(logits.shape), (15, 2, 5))

    def test_training_evaluation_and_prediction_cli_on_synthetic_iam_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            output = Path(temp) / "experiment"
            lines = []
            for group in range(3):
                form = f"a01-{group:03d}u"
                folder = root / "words/a01" / form
                folder.mkdir(parents=True)
                for number in range(2):
                    word_id = f"{form}-00-{number:02d}"
                    Image.new("L", (50, 22), 255).save(folder / (word_id + ".png"))
                    lines.append(f"{word_id} ok 0 0 0 0 0 NN a")
            (root / "words.txt").write_text("\n".join(lines), encoding="utf8")
            with contextlib.redirect_stdout(io.StringIO()):
                htr.main(["train", "--data-root", str(root), "--output", str(output),
                          "--epochs", "1", "--max-samples", "6", "--hidden", "16",
                          "--width", "64", "--batch-size", "2"])
                htr.main(["evaluate", "--data-root", str(root), "--checkpoint", str(output / "best.pt"),
                          "--split-file", str(output / "splits.json")])
                htr.main(["predict", str(folder / (word_id + ".png")),
                          "--checkpoint", str(output / "best.pt")])
            splits = json.loads((output / "splits.json").read_text())
            self.assertEqual(sum(map(len, splits.values())), 6)
            self.assertTrue((output / "config.json").is_file())
            self.assertEqual(json.loads((output / "test-metrics.json").read_text())["samples"], 2)

    def test_synthetic_training_checkpoint_and_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            with contextlib.redirect_stdout(io.StringIO()): htr.run_smoke(Path(temp))
            result = json.loads((Path(temp) / "smoke-result.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["checkpoint_round_trip"])
            self.assertFalse(result["benchmark_accuracy_verified"])


if __name__ == "__main__":
    unittest.main()
