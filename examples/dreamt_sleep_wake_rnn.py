"""DREAMT sleep/wake detection with PyHealth's RNN: a task-configuration ablation.

Authors: Andrae Allen, Hita Pullur
Paper: Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel
    Methods for a Population with Sleep Disorders (Wang et al., CHIL 2024)
Paper link: https://proceedings.mlr.press/v248/wang24a.html
Description: Channel x context-window ablation of SleepWakeDREAMT with
    PyHealth's RNN.

Purpose
-------
Shows how to train and evaluate a PyHealth model on the ``SleepWakeDREAMT``
task, and how two task settings change sleep/wake detection: the sensor
channels used and the temporal context window. It reproduces the epoch-level
wake (positive) vs sleep target of:

    Wang, W. K., et al. "Addressing Wearable Sleep Tracking Inequity: A New
    Dataset and Novel Methods for a Population with Sleep Disorders." CHIL 2024,
    PMLR 248:380-396. https://proceedings.mlr.press/v248/wang24a.html

Ablation design
---------------
Six runs: ``context_window`` k in {0, 2, 5} epochs (30 s, 2.5 min and 5.5 min of
signal around the target epoch) x channels in {ACC only (ACC_X, ACC_Y, ACC_Z),
all seven default E4 channels (ACC_X/Y/Z, HR, TEMP, EDA, BVP)}. Every run uses
the same patient-level 70/10/20 train/validation/test split and seed, the same
PyHealth ``RNN`` and the same training budget; only the task changes. Features
are z-scored with statistics fitted on the training patients. The model with
the best validation AUROC is evaluated on the test patients with F1, AUROC and
Cohen's kappa, all for the wake class.

The task's ``features`` are one flat vector per epoch, so ``RNN`` sees each
sample as a sequence of length 1: in effect it is a feed-forward classifier on
the concatenated features of the 2k + 1 epochs, and a larger k widens its
input layer rather than lengthening the sequence. The ablation therefore
measures how much a classifier gains from neighbouring-epoch features, not
whether a GRU learns across epochs. Context also uses up to 2.5 minutes of
signal after the target epoch, so this is offline (retrospective) scoring.

Why this ablation is novel
--------------------------
- Channels: the paper trains every model on all E4 sensors and never compares
  sensor subsets, although its motivation is that existing wearable datasets
  use a single sensor (the accelerometer). ACC only vs all channels tests that
  claim directly.
- Context window: the paper's temporal context is fixed (Gaussian smoothing,
  moving-window variance, and LSTM post-processing switched on or off in its
  Table 2). It never varies the size of the input window. Varying k asks how
  much context this model needs.

Differences from the paper: simple per-epoch statistics instead of engineered
features, an RNN instead of boosted trees with random effects, one
patient-level split instead of 5-fold cross-validation, no exclusion of
participants with > 20% artifact epochs, and (optionally) a subset of
participants. The numbers are therefore not directly comparable to Table 2.

How to run (from the PyHealth repository root)
----------------------------------------------
Synthetic data (no download; about a minute on a laptop CPU)::

    python examples/dreamt_sleep_wake_rnn.py

Real data (DREAMT 2.1.0 from https://physionet.org/content/dreamt/, signed DUA).
A quick check on 20 participants, then the full run reported below::

    python examples/dreamt_sleep_wake_rnn.py --root /path/to/dreamt/2.1.0 \\
        --max-patients 20
    python examples/dreamt_sleep_wake_rnn.py --root /path/to/dreamt/2.1.0 \\
        --num-workers 8 --results-md results.md

``--max-patients N`` uses the first N participant IDs (sorted) that have a
64 Hz file. Every configuration must produce the same samples (same counts
and labels per split), so the six runs are paired; the script stops if they
differ. Use ``--help`` for all options.

Results
-------
Real DREAMT 2.1.0, all 100 participants (70/9/21 train/val/test patients;
55,894/8,073/16,223 epochs; 21.4% of test epochs are wake), default
arguments except ``--num-workers 8``, seed 42, CPU, about 7 minutes in total:

=====  ========  =====  =====  =======
k      channels  F1     AUROC  kappa
=====  ========  =====  =====  =======
0      ACC only  0.390  0.719  0.298
0      all       0.432  0.764  0.313
2      ACC only  0.507  0.780  0.397
2      all       0.517  0.805  0.397
5      ACC only  0.511  0.808  0.406
5      all       0.529  0.815  0.418
=====  ========  =====  =====  =======

Findings
--------
- Context helps most. Going from k=0 to k=5 raises F1 by 0.12 (ACC only) and
  0.10 (all channels), kappa by about 0.1, and AUROC by 0.089 and 0.051.
- The AUROC advantage of all channels over the accelerometer shrinks as
  context grows: +0.045 at k=0, +0.025 at k=2 and +0.007 at k=5.
- At k=5 the two channel sets are within 0.02 on every metric. A rerun that
  changed only 0.2% of the labels reversed two of the three gaps (AUROC and
  kappa), so treat them as a tie. The all-channel model also has 2.3 times as many
  inputs (308 vs 132) for the same model size and training budget.
- These are single-split results for 21 test participants, without
  confidence intervals, and all runs are well below the paper's best
  results (F1 0.823, AUROC 0.926, kappa 0.694), as expected from the simpler
  features, the model and the single split.
"""

import argparse
import logging
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import DREAMTDataset, PatientSplit, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import SleepWakeDREAMT
from pyhealth.trainer import Trainer

logger = logging.getLogger("dreamt_sleep_wake_rnn")

ACC_CHANNELS = ["ACC_X", "ACC_Y", "ACC_Z"]
CHANNEL_SETS = {
    "ACC only": ACC_CHANNELS,
    "all": list(SleepWakeDREAMT.DEFAULT_CHANNELS),
}
CONTEXT_WINDOWS = (0, 2, 5)
SPLIT_RATIOS = (0.7, 0.1, 0.2)
METRICS = ["f1", "roc_auc", "cohen_kappa"]
EPOCH_ROWS = 30 * 64


def parse_args() -> argparse.Namespace:
    """Parses the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        help="DREAMT version folder (e.g. .../dreamt/2.1.0). Omit to generate a "
        "synthetic DREAMT-style dataset.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Use only the first N participants (sorted IDs with a 64 Hz file).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Processes for set_task."
    )
    parser.add_argument(
        "--synthetic-patients", type=int, default=20, help="Synthetic mode only."
    )
    parser.add_argument(
        "--synthetic-epochs", type=int, default=60, help="Synthetic mode only."
    )
    parser.add_argument(
        "--output-dir",
        help="Where the Trainer writes checkpoints and logs (default: a temp dir).",
    )
    parser.add_argument(
        "--results-md", help="Also write the results table to this Markdown file."
    )
    args = parser.parse_args()
    if args.max_patients is not None and args.max_patients < 1:
        parser.error("--max-patients must be at least 1")
    if args.synthetic_epochs < 20:
        parser.error("--synthetic-epochs must be at least 20")
    if args.synthetic_patients < 1:
        parser.error("--synthetic-patients must be at least 1")
    return args


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch.

    Args:
        seed: the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_stages(n_epochs: int, rng: np.random.Generator) -> list[str]:
    """Generates a plausible night of sleep stages, one per 30-s epoch.

    Every night starts with preparation (P) and wake, has a mid-night
    awakening and a missing epoch, and ends awake, so every patient has both
    classes and any patient-level split is usable.

    Args:
        n_epochs: number of epochs (at least 20).
        rng: random generator.

    Returns:
        The stage of each epoch.
    """
    stages = ["P", "P"] + ["W"] * int(rng.integers(3, 6))
    while len(stages) < n_epochs - 4:
        stage = rng.choice(["N1", "N2", "N2", "N3", "R"])
        stages += [str(stage)] * int(rng.integers(2, 6))
        if rng.random() < 0.15:
            stages += ["W"] * int(rng.integers(1, 3))
    stages = stages[: n_epochs - 4]
    stages[len(stages) // 2] = "Missing"
    return stages + ["W"] * 4


def make_synthetic_dreamt(
    root: Path, n_patients: int, n_epochs: int, seed: int
) -> Path:
    """Writes a synthetic dataset in the DREAMT 2.x folder layout.

    Wake epochs have more wrist movement and a higher heart rate than sleep
    epochs, so the task is learnable but not trivial. The numbers mean
    nothing clinically; they only exercise the pipeline.

    Args:
        root: an empty directory to write into.
        n_patients: number of synthetic participants.
        n_epochs: epochs per participant.
        seed: random seed.

    Returns:
        ``root``, which holds ``participant_info.csv`` and ``data_64Hz/``.
    """
    rng = np.random.default_rng(seed)
    (root / "data_64Hz").mkdir(parents=True)
    sids = [f"S{i:03d}" for i in range(1, n_patients + 1)]
    pd.DataFrame(
        {
            "SID": sids,
            "AGE": rng.integers(21, 87, n_patients),
            "GENDER": rng.choice(["M", "F"], n_patients),
            "BMI": rng.uniform(20, 45, n_patients).round(1),
            "OAHI": rng.uniform(0, 40, n_patients).round(1),
            "AHI": rng.uniform(0, 60, n_patients).round(1),
            "Mean_SaO2": [f"{v}%" for v in rng.integers(88, 99, n_patients)],
            "Arousal Index": rng.uniform(5, 60, n_patients).round(1),
            "MEDICAL_HISTORY": "synthetic",
            "Sleep_Disorders": "synthetic",
        }
    ).to_csv(root / "participant_info.csv", index=False)

    for sid in sids:
        stages = synthetic_stages(n_epochs, rng)
        wake = np.repeat([s == "W" for s in stages], EPOCH_ROWS)
        n_rows = len(wake)
        movement = np.where(wake, 0.3, 0.2) * rng.uniform(0.7, 1.3, n_rows)
        data = {"TIMESTAMP": np.arange(n_rows) / 64}
        for axis in ACC_CHANNELS:
            data[axis] = rng.normal(0, movement) + rng.normal(0, 0.05)
        data["HR"] = 60 + 3 * wake + rng.normal(0, 6, n_rows)
        data["TEMP"] = 34 - 0.3 * wake + rng.normal(0, 0.1, n_rows)
        data["EDA"] = np.abs(0.2 + 0.05 * wake + rng.normal(0, 0.05, n_rows))
        data["BVP"] = rng.normal(0, 30, n_rows)
        data["IBI"] = 60 / data["HR"]
        data["Sleep_Stage"] = np.repeat(stages, EPOCH_ROWS)
        frame = pd.DataFrame(data).round(4)
        frame.to_csv(root / "data_64Hz" / f"{sid}_whole_df.csv", index=False)
    return root


def make_subset_root(root: Path, max_patients: int, workdir: Path) -> Path:
    """Builds a DREAMT folder holding only the first ``max_patients``.

    ``DREAMTDataset`` reads every participant listed in
    ``participant_info.csv``, so the subset folder gets a filtered copy of
    that file and symlinks to the chosen 64 Hz recordings.

    Args:
        root: the full DREAMT version folder.
        max_patients: how many participants to keep.
        workdir: an empty directory for the subset.

    Returns:
        The subset folder, to pass to ``DREAMTDataset``.

    Raises:
        ValueError: if fewer than ``max_patients`` participants have a file.
    """
    info = pd.read_csv(root / "participant_info.csv")
    available = sorted(
        sid
        for sid in info["SID"]
        if (root / "data_64Hz" / f"{sid}_whole_df.csv").is_file()
    )
    if len(available) < max_patients:
        raise ValueError(
            f"Only {len(available)} participants have a 64 Hz file, "
            f"{max_patients} requested"
        )
    chosen = available[:max_patients]
    (workdir / "data_64Hz").mkdir(parents=True)
    info[info["SID"].isin(chosen)].to_csv(workdir / "participant_info.csv", index=False)
    for sid in chosen:
        name = f"{sid}_whole_df.csv"
        (workdir / "data_64Hz" / name).symlink_to((root / "data_64Hz" / name).resolve())
    logger.info("Using %d participants: %s", len(chosen), ", ".join(chosen))
    return workdir


def label_counts(dataset) -> tuple[int, int]:
    """Counts the samples and wake labels of a sample dataset.

    Args:
        dataset: a ``SampleDataset`` returned by ``set_task``.

    Returns:
        ``(number of samples, number of wake samples)``.
    """
    labels = [b["label"] for b in get_dataloader(dataset, batch_size=4096)]
    if not labels:
        return 0, 0
    labels = torch.cat(labels).flatten()
    return int(labels.numel()), int(labels.sum().item())


def run_config(
    dataset: DREAMTDataset,
    context_window: int,
    channel_set: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    """Builds the task samples, trains the RNN and evaluates it on the test set.

    Args:
        dataset: the loaded DREAMT dataset.
        context_window: neighbouring epochs on each side (k).
        channel_set: a key of ``CHANNEL_SETS``.
        args: the command-line arguments.
        output_dir: where the Trainer writes checkpoints.

    Returns:
        One row of the results table.

    Raises:
        RuntimeError: if the validation or test split has only one class.
    """
    task = SleepWakeDREAMT(
        context_window=context_window, channels=CHANNEL_SETS[channel_set]
    )
    # The task cache is keyed by the task's settings, not its code; clear it so
    # an edited task is never served stale samples.
    dataset.clear_task_cache(task)
    start = time.time()
    train, val, test = dataset.set_task(
        task,
        split=PatientSplit(ratios=SPLIT_RATIOS, seed=args.seed),
        num_workers=args.num_workers,
    )
    task_seconds = time.time() - start

    counts = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        n, n_wake = label_counts(part)
        counts[name] = (n, n_wake)
        logger.info(
            "k=%d %s %s: %d samples, %d wake (%.1f%%)",
            context_window,
            channel_set,
            name,
            n,
            n_wake,
            100 * n_wake / max(n, 1),
        )
        if name != "train" and (n_wake == 0 or n_wake == n):
            raise RuntimeError(
                f"The {name} split has only one class; use more patients"
            )

    seed_everything(args.seed)
    model = RNN(
        dataset=train, embedding_dim=args.hidden_dim, hidden_dim=args.hidden_dim
    )
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device="cpu",
        output_path=str(output_dir),
        exp_name=f"k{context_window}_{channel_set.replace(' ', '_')}",
    )
    start = time.time()
    trainer.train(
        train_dataloader=get_dataloader(
            train, batch_size=args.batch_size, shuffle=True, seed=args.seed
        ),
        val_dataloader=get_dataloader(val, batch_size=1024),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="roc_auc",
    )
    scores = trainer.evaluate(get_dataloader(test, batch_size=1024))
    return {
        "k": context_window,
        "channels": channel_set,
        "n_features": task.feature_dim,
        "samples (train/val/test)": "/".join(str(n) for n, _ in counts.values()),
        "F1": scores["f1"],
        "AUROC": scores["roc_auc"],
        "kappa": scores["cohen_kappa"],
        "set_task s": task_seconds,
        "train s": time.time() - start,
        "_counts": counts,
    }


def results_table(rows: list[dict]) -> str:
    """Formats the results as a Markdown table.

    Args:
        rows: the dicts returned by ``run_config``.

    Returns:
        The table.
    """
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Runs the six-configuration ablation and prints the results table."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed_everything(args.seed)
    workdir = Path(tempfile.mkdtemp(prefix="sleep_wake_dreamt_"))
    output_dir = Path(args.output_dir) if args.output_dir else workdir / "output"
    synthetic = args.root is None
    try:
        if synthetic:
            root = make_synthetic_dreamt(
                workdir / "synthetic",
                args.synthetic_patients,
                args.synthetic_epochs,
                args.seed,
            )
            logger.info("Synthetic DREAMT-style data written to %s", root)
        else:
            root = Path(args.root)
        if args.max_patients is not None:
            root = make_subset_root(root, args.max_patients, workdir / "subset")

        dataset = DREAMTDataset(root=str(root))
        rows = []
        reference = None
        for context_window in CONTEXT_WINDOWS:
            for channel_set in CHANNEL_SETS:
                row = run_config(dataset, context_window, channel_set, args, output_dir)
                counts = row.pop("_counts")
                # Paired comparison: a recording skipped by one configuration
                # (e.g. a missing channel) would change the samples.
                if reference is not None and counts != reference:
                    raise RuntimeError(
                        f"k={context_window} {channel_set} has samples (n, wake) "
                        f"{counts}, but the first run had {reference}; check the "
                        "warnings for skipped recordings"
                    )
                reference = counts
                rows.append(row)
                logger.info("Done: %s", row)

        title = "synthetic data" if synthetic else f"DREAMT at {args.root}"
        table = results_table(rows)
        print(f"\n## Results on {title} (test set, seed {args.seed})\n\n{table}\n")
        if args.results_md:
            Path(args.results_md).write_text(
                f"# SleepWakeDREAMT ablation ({title})\n\n{table}\n"
            )
            logger.info("Wrote %s", args.results_md)
        if synthetic or args.max_patients is not None:
            # Its cache belongs to a throwaway folder; don't leave it behind.
            shutil.rmtree(dataset.cache_dir, ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
