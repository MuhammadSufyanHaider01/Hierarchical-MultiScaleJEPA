from __future__ import annotations

import sys
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from phase5_exits import _train_exit  # noqa: E402
from phase5 import TaskSpec  # noqa: E402
from models import Encoder  # noqa: E402


class Phase5ExitStoppingTest(unittest.TestCase):
    def test_validation_metric_stopping_preserves_the_first_best_head(self):
        torch.manual_seed(9)
        encoder = Encoder(
            img_size=(32, 32),
            patch_size=8,
            embed_dim=16,
            depth=8,
            num_heads=2,
        )
        samples = torch.zeros(4, 1, 32, 32)
        labels = torch.zeros(4, dtype=torch.long)
        loader = DataLoader(TensorDataset(samples, labels), batch_size=2, shuffle=False)
        spec = TaskSpec(
            name="toy",
            path="",
            target_type="classification",
            num_outputs=2,
            higher_is_better=True,
            metric_name="macro_accuracy",
            sample_shape=(1, 32, 32),
        )
        args = SimpleNamespace(
            learning_rate=0.0,
            weight_decay=0.0,
            limit_train_batches=None,
            limit_validation_batches=None,
            epochs=5,
            head_early_stopping_min_delta=0.0,
            head_early_stopping_patience=1,
            disable_head_early_stopping=False,
            progress_every_n_batches=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            log = io.StringIO()
            with redirect_stdout(log):
                result = _train_exit(
                    encoder,
                    exit_layer=2,
                    spec=spec,
                    train_loader=loader,
                    validation_loader=loader,
                    args=args,
                    device=torch.device("cpu"),
                    head_seed=7,
                    head_path=Path(directory) / "head.pt",
                )

        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_completed"], 2)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(log.getvalue().count("train_loss="), result["epochs_completed"])
        self.assertEqual(log.getvalue().count("\r"), result["epochs_completed"])


if __name__ == "__main__":
    unittest.main()
