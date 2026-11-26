"""Tests for the SleepWakeDREAMT task, on 5 synthetic DREAMT-style patients.

Authors: Andrae Allen, Hita Pullur
Paper: Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel
    Methods for a Population with Sleep Disorders (Wang et al., CHIL 2024)
Paper link: https://proceedings.mlr.press/v248/wang24a.html
Description: Unit tests for SleepWakeDREAMT on synthetic data.

The 64 Hz CSVs are tiny synthetic files in a temporary directory. Patients are
built directly as :class:`pyhealth.data.Patient` objects with the same columns
that :class:`~pyhealth.datasets.DREAMTDataset` produces, so the tests exercise
the real Patient/Event plumbing without the cost of building the dataset. AHI
and BMI are stored as strings (or null), as the dataset delivers them.
"""

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import SleepWakeDREAMT

EPOCH = 1920  # rows per 30-second epoch at 64 Hz
COLUMNS = ["TIMESTAMP", "BVP", "IBI", "EDA", "TEMP"]
COLUMNS += ["ACC_X", "ACC_Y", "ACC_Z", "HR", "Sleep_Stage"]
N_DEFAULT = len(SleepWakeDREAMT.DEFAULT_CHANNELS) * 4  # 28 features per epoch

# patient id -> (stage per epoch, extra rows, AHI, BMI)
PATIENTS = {
    "S001": (["W", "N1", "N2", "N3", "R", "W"], 0, 2.0, 24.0),
    "S002": (["P", "W", "Missing", "N2", "N2", "W"], 0, 10.0, 31.0),
    "S003": (["W", "W", "W", "W"], 500, 20.0, 35.0),
    "S004": ([], 1000, 40.0, 28.0),
    "S005": ([], 0, None, None),
}
# S002 starts with 700 extra P rows, so its PSG grid is offset from row 0.
LEAD_ROWS = {"S002": 700}


def write_recording(
    path: Path, stages: list[str], extra_rows: int, seed: int, lead_rows: int = 0
) -> None:
    """Writes a synthetic 64 Hz CSV with a constant stage per epoch.

    Args:
        path: where to write the CSV.
        stages: the Sleep_Stage of each full epoch.
        extra_rows: rows appended after the last full epoch (stage "W").
        seed: seed for the random signal values.
        lead_rows: rows of the first stage written before the first epoch.
    """
    rng = np.random.default_rng(seed)
    n_rows = lead_rows + len(stages) * EPOCH + extra_rows
    data = {name: rng.normal(size=n_rows) for name in COLUMNS[:-1]}
    data["TIMESTAMP"] = np.arange(n_rows) / 64
    lead = [stages[0]] * lead_rows if lead_rows else []
    data["Sleep_Stage"] = lead + list(np.repeat(stages, EPOCH)) + ["W"] * extra_rows
    frame = pd.DataFrame(data, columns=COLUMNS)
    if path.stem.startswith("S003"):
        frame["ACC_X"] = frame["ACC_X"].astype(object)
        frame.loc[10, "ACC_X"] = "bad"  # a stray non-numeric value
    frame.to_csv(path, index=False)


def as_text(value: float | None) -> str | None:
    """Formats a metadata number as DREAMTDataset stores it (e.g. "31.0").

    Args:
        value: the number, or None.

    Returns:
        The text, or None.
    """
    return None if value is None else str(float(value))


def make_patient(patient_id: str, file_64hz: str | None, ahi, bmi) -> Patient:
    """Builds a Patient with one dreamt_sleep event, as DREAMTDataset does.

    Args:
        patient_id: the participant ID.
        file_64hz: path of the 64 Hz CSV, or None.
        ahi: the AHI, or None; stored as text like the dataset's metadata.
        bmi: the BMI, or None; stored as text like the dataset's metadata.

    Returns:
        The patient.
    """
    return Patient(
        patient_id=patient_id,
        data_source=pl.DataFrame(
            {
                "patient_id": [patient_id],
                "event_type": ["dreamt_sleep"],
                "timestamp": pl.Series([None], dtype=pl.Datetime),
                "_source_row": [0],
                "dreamt_sleep/file_64hz": pl.Series([file_64hz], dtype=pl.Utf8),
                "dreamt_sleep/ahi": pl.Series([as_text(ahi)], dtype=pl.Utf8),
                "dreamt_sleep/bmi": pl.Series([as_text(bmi)], dtype=pl.Utf8),
            }
        ),
    )


class TestSleepWakeDREAMT(unittest.TestCase):
    """Sample processing, labels, features and edge cases of SleepWakeDREAMT."""

    @classmethod
    def setUpClass(cls) -> None:
        """Writes the 5 synthetic recordings and runs each task config once."""
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.files = {}
        cls.patients = {}
        for seed, (pid, (stages, extra, ahi, bmi)) in enumerate(PATIENTS.items()):
            path = root / f"{pid}_whole_df.csv"
            if pid == "S005":
                pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
            else:
                write_recording(path, stages, extra, seed, LEAD_ROWS.get(pid, 0))
            cls.files[pid] = path
            cls.patients[pid] = make_patient(pid, str(path), ahi, bmi)

        # S004 and S005 log expected warnings; keep them out of the output.
        task_logger = logging.getLogger("pyhealth.tasks.sleep_wake_dreamt")
        previous_level = task_logger.level
        task_logger.setLevel(logging.ERROR)
        try:
            task = SleepWakeDREAMT()
            cls.default = {pid: task(p) for pid, p in cls.patients.items()}
        finally:
            task_logger.setLevel(previous_level)
        cls.context = SleepWakeDREAMT(context_window=2)(cls.patients["S001"])
        cls.context_s002 = SleepWakeDREAMT(context_window=1)(cls.patients["S002"])
        cls.subset = SleepWakeDREAMT(channels=["ACC_X", "HR"])(cls.patients["S001"])

    @classmethod
    def tearDownClass(cls) -> None:
        """Deletes the temporary directory."""
        cls.tmp.cleanup()

    def test_sample_counts(self) -> None:
        """One sample per labelled full epoch; short and empty files give none."""
        counts = {pid: len(samples) for pid, samples in self.default.items()}
        self.assertEqual(
            counts, {"S001": 6, "S002": 4, "S003": 4, "S004": 0, "S005": 0}
        )

    def test_label_mapping(self) -> None:
        """W maps to 1 and N1/N2/N3/R map to 0."""
        labels = [s["label"] for s in self.default["S001"]]
        self.assertEqual(labels, [1, 0, 0, 0, 0, 1])

    def test_missing_and_preparation_dropped(self) -> None:
        """P and Missing epochs are not samples but keep their epoch index."""
        samples = self.default["S002"]
        self.assertEqual([s["epoch_index"] for s in samples], [1, 3, 4, 5])
        self.assertEqual([s["label"] for s in samples], [1, 0, 0, 1])

    def test_feature_length_and_keys(self) -> None:
        """Defaults give 7 channels x 4 statistics; samples carry all keys."""
        sample = self.default["S001"][0]
        self.assertEqual(len(sample["features"]), N_DEFAULT)
        self.assertEqual(SleepWakeDREAMT().feature_dim, N_DEFAULT)
        self.assertEqual(
            sorted(sample),
            ["apnea_severity", "epoch_index", "features", "label"]
            + ["obesity", "patient_id"],
        )
        self.assertEqual(sample["patient_id"], "S001")

    def test_context_window_padding(self) -> None:
        """k=2 concatenates 5 epochs and repeats the edge epochs as padding."""
        first, last = self.context[0]["features"], self.context[-1]["features"]
        self.assertEqual(len(first), N_DEFAULT * 5)
        blocks = np.reshape(first, (5, N_DEFAULT))
        own = self.default["S001"][0]["features"]
        # Blocks 0-2 are all epoch 0 (two repeats, then the centre block).
        for block in blocks[:3]:
            np.testing.assert_allclose(block, own)
        np.testing.assert_allclose(blocks[3], self.default["S001"][1]["features"])
        last_blocks = np.reshape(last, (5, N_DEFAULT))
        for block in last_blocks[2:]:
            np.testing.assert_allclose(block, self.default["S001"][-1]["features"])

    def test_unlabelled_epochs_give_context(self) -> None:
        """S002's first target (epoch 1) sees the P and Missing epochs."""
        sample = self.context_s002[0]
        self.assertEqual(sample["epoch_index"], 1)
        blocks = np.reshape(sample["features"], (3, N_DEFAULT))
        recording = pd.read_csv(self.files["S002"])
        start = LEAD_ROWS["S002"]
        for block, epoch in zip(blocks, [0, 1, 2]):  # P, W, Missing
            first_row = start + epoch * EPOCH
            rows = recording["ACC_X"].to_numpy()[first_row:][:EPOCH]
            self.assertAlmostEqual(block[0], rows.mean(), places=5)

    def test_epochs_align_to_psg_grid(self) -> None:
        """Epochs start at the first stage change, not at row 0."""
        recording = pd.read_csv(self.files["S002"])
        start = LEAD_ROWS["S002"] + EPOCH  # epoch 1, the first W epoch
        end = start + EPOCH
        expected = recording["ACC_X"].to_numpy()[start:end].mean()
        sample = self.default["S002"][0]
        self.assertAlmostEqual(sample["features"][0], expected, places=5)

    def test_mixed_stage_plurality(self) -> None:
        """Epochs take the most common wake/sleep/unscored class; ties drop."""
        task = SleepWakeDREAMT()

        def epoch(*parts: tuple[str, int]) -> list[str]:
            return [stage for stage, rows in parts for _ in range(rows)]

        stages = np.array(
            epoch(("W", 640), ("N1", 640), ("N3", 640))  # sleep 2/3
            + epoch(("W", 768), ("N2", 672), ("P", 480))  # wake 40%: plurality
            + epoch(("W", 960), ("N2", 960)),  # a tie
            dtype=object,
        )
        self.assertEqual(task._epoch_labels(stages, 3).tolist(), [0, 1, -1])

    def test_grid_segments(self) -> None:
        """A one-row glitch can't move the grid; a confirmed shift starts a segment."""
        task = SleepWakeDREAMT()

        def night(lead: int, blocks: list[tuple[str, int]]) -> np.ndarray:
            rows = ["P"] * lead + [s for s, n in blocks for _ in range(n)]
            return np.array(rows, dtype=object)

        five = [(s, EPOCH) for s in ["W", "N2", "W", "N2", "W"]]
        clean = night(700, five)
        self.assertEqual(task._grid_segments(clean), [(700, 5)])
        glitch = clean.copy()
        glitch[100] = "Missing"  # an isolated one-row label
        self.assertEqual(task._grid_segments(glitch), [(700, 5)])
        # Three epochs, a 960-row gap, then four epochs on a shifted grid.
        shifted = night(
            700,
            [(s, EPOCH) for s in ["W", "N2", "W"]]
            + [("Missing", 960)]
            + [(s, EPOCH) for s in ["N2", "W", "N2", "W"]],
        )
        self.assertEqual(task._grid_segments(shifted), [(700, 3), (7420, 4)])
        self.assertEqual(task._grid_segments(np.array(["W"] * 5000)), [(0, 2)])

    def test_non_finite_readings_ignored(self) -> None:
        """inf and values beyond float32 are ignored like NaN."""
        task = SleepWakeDREAMT(channels=["HR"])
        signals = np.full((EPOCH, 1), 2.0)
        signals[0, 0] = np.inf
        signals[1, 0] = 1e39
        stats = task._epoch_features(signals, 1)[0]
        np.testing.assert_allclose(stats, [2.0, 0.0, 2.0, 2.0])

    def test_non_numeric_value_becomes_nan(self) -> None:
        """A stray text value in S003 doesn't stop its samples."""
        samples = self.default["S003"]
        self.assertEqual(len(samples), 4)
        self.assertTrue(np.isfinite(samples[0]["features"]).all())

    def test_channel_subset_values(self) -> None:
        """Statistics follow the channel order: mean, std (ddof=0), min, max."""
        self.assertEqual(len(self.subset[0]["features"]), 8)
        epoch0 = pd.read_csv(self.files["S001"], nrows=EPOCH)
        expected = []
        for channel in ["ACC_X", "HR"]:
            x = epoch0[channel].to_numpy()
            expected += [x.mean(), x.std(), x.min(), x.max()]
        np.testing.assert_allclose(self.subset[0]["features"], expected, rtol=1e-5)

    def test_invalid_arguments(self) -> None:
        """Bad window sizes, epoch settings and channel lists are rejected."""
        with self.assertRaises(ValueError):
            SleepWakeDREAMT(context_window=-1)
        with self.assertRaises(ValueError):
            SleepWakeDREAMT(channels=[])
        with self.assertRaises(ValueError):
            SleepWakeDREAMT(channels=["FOO"])
        with self.assertRaises(ValueError):
            SleepWakeDREAMT(channels=["HR", "HR"])
        with self.assertRaises(TypeError):
            SleepWakeDREAMT(context_window=1.5)
        with self.assertRaises(TypeError):
            SleepWakeDREAMT(channels="HR")
        with self.assertRaises(TypeError):
            SleepWakeDREAMT(channels={"HR", "ACC_X"})  # unordered

    def test_group_attributes(self) -> None:
        """Apnea severity and obesity come from AHI and BMI (numbers or text)."""
        groups = {
            pid: (
                self.default[pid][0]["apnea_severity"],
                self.default[pid][0]["obesity"],
            )
            for pid in ["S001", "S002", "S003"]
        }
        self.assertEqual(
            groups,
            {"S001": ("none", 0), "S002": ("mild", 1), "S003": ("moderate", 1)},
        )
        self.assertEqual(SleepWakeDREAMT.apnea_severity_from_ahi(40), "severe")
        self.assertEqual(SleepWakeDREAMT.apnea_severity_from_ahi(None), "unknown")
        self.assertEqual(SleepWakeDREAMT.apnea_severity_from_ahi("n/a"), "unknown")
        self.assertEqual(SleepWakeDREAMT.apnea_severity_from_ahi(15), "moderate")
        self.assertEqual(SleepWakeDREAMT.obesity_from_bmi(None), -1)
        self.assertEqual(SleepWakeDREAMT.obesity_from_bmi(29.9), 0)

    def test_missing_file_returns_empty(self) -> None:
        """A path that doesn't exist, or no path, gives [] and a warning."""
        task = SleepWakeDREAMT()
        for path in [str(Path(self.tmp.name) / "S999_whole_df.csv"), None]:
            patient = make_patient("S999", path, 1.0, 20.0)
            with self.assertLogs("pyhealth.tasks.sleep_wake_dreamt", "WARNING"):
                self.assertEqual(task(patient), [])


if __name__ == "__main__":
    unittest.main()
