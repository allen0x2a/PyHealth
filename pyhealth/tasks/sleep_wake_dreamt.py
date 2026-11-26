"""Sleep/wake classification on DREAMT wearable data.

Authors: Andrae Allen, Hita Pullur
Paper: Addressing Wearable Sleep Tracking Inequity: A New Dataset and Novel
    Methods for a Population with Sleep Disorders (Wang et al., CHIL 2024)
Paper link: https://proceedings.mlr.press/v248/wang24a.html
Description: Epoch-level sleep/wake detection task (SleepWakeDREAMT) for
    DREAMTDataset.

This module defines :class:`SleepWakeDREAMT`, a standalone task for
:class:`~pyhealth.datasets.DREAMTDataset`. It turns each participant's 64 Hz
Empatica E4 recording into one sample per labelled 30-second epoch, with simple
per-channel statistics as features and wake (1) vs sleep (0) as the label,
following the epoch-level sleep/wake detection setting of:

    Wang, W. K., et al. "Addressing Wearable Sleep Tracking Inequity: A New
    Dataset and Novel Methods for a Population with Sleep Disorders."
    Proceedings of the Conference on Health, Inference, and Learning (CHIL),
    PMLR 248:380-396, 2024. https://proceedings.mlr.press/v248/wang24a.html
"""

import logging
import math
import os
import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..data import Patient
from .base_task import BaseTask

logger = logging.getLogger(__name__)


class SleepWakeDREAMT(BaseTask):
    """Epoch-level sleep/wake detection from DREAMT wrist-worn sensor data.

    DREAMT (Wang et al., CHIL 2024) pairs an Empatica E4 wristband with
    gold-standard polysomnography for 100 adults seen at a sleep disorders
    clinic. This task reproduces the paper's binary target, wake (positive)
    vs sleep, at the level of 30-second epochs, using simple statistics of the
    raw 64 Hz signals so that any PyHealth model can be trained on it. It runs
    on :class:`~pyhealth.datasets.DREAMTDataset`. Paper:
    https://proceedings.mlr.press/v248/wang24a.html

    How ``__call__`` builds the samples for one patient:

    1. **Read** the patient's single ``dreamt_sleep`` event and load its
       ``file_64hz`` CSV, keeping only the selected ``channels`` and
       ``Sleep_Stage`` (:meth:`_read_recording`). Non-numeric signal values
       become NaN. A missing, unreadable, empty or shorter-than-one-epoch
       file logs a warning and yields ``[]``.
    2. **Epoch**: cut the recording into non-overlapping 30-second epochs
       (1920 rows at 64 Hz) aligned to the PSG scoring grid
       (:meth:`_grid_segments`). PSG stages change only on scored-epoch
       boundaries, but the file does not start on one, and in a few DREAMT
       recordings the grid shifts after a gap. A stage change starts a grid
       segment when the next stage change confirms its offset, so a stray
       one-row label cannot move the grid. Rows outside whole epochs (before
       the first boundary, at a shift, and at the end) are dropped.
    3. **Features**: for every epoch and channel compute mean, standard
       deviation (population, ``ddof=0``), min and max, ignoring NaN,
       infinite and out-of-float32-range readings; a statistic that is still
       NaN (no usable reading) becomes 0 (:meth:`_epoch_features`).
    4. **Label**: map each row's stripped ``Sleep_Stage`` to wake (``W``),
       sleep (``N1``/``N2``/``N3``/``R``) or unscored (``P`` for
       preparation, ``Missing`` or anything else), and give each epoch its
       most common class, a plurality vote (:meth:`_epoch_labels`). Wake is
       1 and sleep is 0. Epochs whose most common class is unscored, or with
       a tie for first, are not emitted as samples. With the grid aligned,
       almost every epoch has a single stage.
    5. **Context**: concatenate the features of epochs ``i - k`` to ``i + k``
       (``k = context_window``), repeating the first and last epoch past the
       start and end of the recording (:meth:`_add_context`). Unlabelled
       neighbours still provide context, since only the centre epoch is the
       target. Neighbours across a grid shift are less than 30 seconds
       apart, so they are treated as adjacent.
    6. **Emit** one sample per labelled epoch with the group attributes used
       as random effects in the paper (:meth:`apnea_severity_from_ahi`,
       :meth:`obesity_from_bmi`).

    The ``features`` vector is flat, of length
    ``len(channels) * 4 * (2 * context_window + 1)``, laid out as
    ``[epoch i-k | ... | epoch i | ... | epoch i+k]``, where each block is
    ``[ch1_mean, ch1_std, ch1_min, ch1_max, ch2_mean, ...]`` in ``channels``
    order. For example, the 7 default channels with ``context_window=2`` give
    ``7 * 4 * 5 = 140`` values: values 56-83 describe the target epoch, and
    for the first epoch of a recording the first 56 repeat it. Edge padding
    keeps the padded values in the range of real data, so they don't become
    outliers after z-scoring. The input schema
    z-scores each position with statistics fitted on the training samples
    (see :class:`~pyhealth.processors.TensorProcessor`).

    Each sample is a dict with:

    - ``patient_id`` (str): the DREAMT participant ID, e.g. ``"S002"``.
    - ``epoch_index`` (int): the epoch's position among all aligned epochs
      of the recording, counting unlabelled ones, so gaps show dropped epochs.
    - ``features`` (list[float]): the vector described above.
    - ``label`` (int): 1 for wake, 0 for sleep.
    - ``apnea_severity`` (str): ``"none"``, ``"mild"``, ``"moderate"``,
      ``"severe"`` or ``"unknown"``, from the participant's AHI.
    - ``obesity`` (int): 1 if BMI >= 30 kg/m^2, 0 if below, -1 if missing.

    Args:
        context_window: number of neighbouring epochs ``k`` on each side to
            concatenate to the target epoch. 0 uses the target epoch only.
        channels: signal columns to use, as an ordered list or tuple: a subset
            of :attr:`DEFAULT_CHANNELS` plus ``"IBI"``. Defaults to
            :attr:`DEFAULT_CHANNELS`.

    Attributes:
        task_name (str): ``"SleepWakeDREAMT"``.
        input_schema (dict[str, Any]): ``features`` as a z-scored ``"tensor"``.
        output_schema (dict[str, str]): ``label`` as ``"binary"``.
        DEFAULT_CHANNELS (list[str]): the seven E4 signals used by default.
        EPOCH_LENGTH (int): rows per epoch, 30 s x 64 Hz = 1920. The task
            reads DREAMT's 64 Hz files, whose PSG labels are scored in
            30-second epochs, so the epoch length is fixed.
        context_window (int): see Args.
        channels (list[str]): the channels in use, in feature order.

    Raises:
        TypeError: if ``context_window`` is not an int, or ``channels`` is not
            a list or tuple of names.
        ValueError: if ``context_window`` is negative, or ``channels`` is
            empty or contains duplicates or an unsupported channel.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset, PatientSplit
        >>> from pyhealth.tasks import SleepWakeDREAMT
        >>> task = SleepWakeDREAMT(context_window=2)
        >>> task.feature_dim
        140
        >>> dataset = DREAMTDataset(root="/path/to/dreamt/2.1.0")  # doctest: +SKIP
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.7, 0.1, 0.2), seed=42)
        ... )
        >>> sorted(train[0])  # doctest: +SKIP
        ['apnea_severity', 'epoch_index', 'features', 'label', 'obesity', 'patient_id']
    """

    task_name: ClassVar[str] = "SleepWakeDREAMT"
    input_schema: ClassVar[dict[str, Any]] = {
        "features": ("tensor", {"normalization": "zscore"})
    }
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    DEFAULT_CHANNELS: ClassVar[list[str]] = [
        "ACC_X",
        "ACC_Y",
        "ACC_Z",
        "HR",
        "TEMP",
        "EDA",
        "BVP",
    ]
    EXTRA_CHANNELS: ClassVar[list[str]] = ["IBI"]
    STAGE_TO_LABEL: ClassVar[dict[str, int]] = {
        "W": 1,
        "N1": 0,
        "N2": 0,
        "N3": 0,
        "R": 0,
    }
    N_STATISTICS: ClassVar[int] = 4
    EPOCH_LENGTH: ClassVar[int] = 30 * 64

    def __init__(
        self,
        context_window: int = 0,
        channels: Sequence[str] | None = None,
    ) -> None:
        """Validates and stores the task configuration.

        Args:
            context_window: neighbouring epochs on each side, >= 0.
            channels: signal columns to use, as a list or tuple; ``None`` for
                the defaults.

        Raises:
            TypeError: if ``context_window`` is not an int, or ``channels`` is
                not a list or tuple.
            ValueError: if any argument value is invalid.
        """
        super().__init__()
        if isinstance(context_window, bool) or not isinstance(context_window, int):
            raise TypeError(f"context_window must be an int, got {context_window!r}")
        if context_window < 0:
            raise ValueError(f"context_window must be >= 0, got {context_window}")
        if channels is None:
            channels = list(self.DEFAULT_CHANNELS)
        if not isinstance(channels, (list, tuple)):
            # A str, set or other iterable would give a wrong or unstable order.
            raise TypeError(
                f"channels must be a list or tuple of names, got {channels!r}"
            )
        channels = list(channels)
        if not channels:
            raise ValueError("channels must contain at least one channel")
        allowed = self.DEFAULT_CHANNELS + self.EXTRA_CHANNELS
        unknown = [c for c in channels if c not in allowed]
        if unknown:
            raise ValueError(f"Unknown channels {unknown}; choose from {allowed}")
        if len(set(channels)) != len(channels):
            raise ValueError(f"channels contains duplicates: {channels}")
        self.context_window = context_window
        self.channels = channels

    @property
    def feature_dim(self) -> int:
        """int: length of each sample's ``features`` vector."""
        return len(self.channels) * self.N_STATISTICS * (2 * self.context_window + 1)

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds one sample per labelled epoch of a patient's recording.

        Args:
            patient: a DREAMT patient with one ``dreamt_sleep`` event whose
                attributes include ``file_64hz``, ``ahi`` and ``bmi``.

        Returns:
            The samples in epoch order (see the class docstring for their
            keys), or ``[]`` if the patient has no usable recording.
        """
        events = patient.get_events(event_type="dreamt_sleep")
        if not events:
            logger.warning("Patient %s has no dreamt_sleep event", patient.patient_id)
            return []
        event = events[0]
        recording = self._read_recording(patient.patient_id, event)
        if recording is None:
            return []

        stages = self._clean_stages(recording["Sleep_Stage"].to_numpy())
        segments = self._grid_segments(stages)
        if not segments:
            logger.warning(
                "Patient %s: no full epoch after aligning to the PSG grid",
                patient.patient_id,
            )
            return []
        signals = recording[self.channels].to_numpy(dtype=np.float64)
        epoch_features, epoch_labels = [], []
        for start, n_epochs in segments:
            rows = slice(start, start + n_epochs * self.EPOCH_LENGTH)
            epoch_features.append(self._epoch_features(signals[rows], n_epochs))
            epoch_labels.append(self._epoch_labels(stages[rows], n_epochs))
        features = self._add_context(np.concatenate(epoch_features))
        labels = np.concatenate(epoch_labels)

        severity = self.apnea_severity_from_ahi(event.attr_dict.get("ahi"))
        obesity = self.obesity_from_bmi(event.attr_dict.get("bmi"))
        samples = []
        for index in np.flatnonzero(labels >= 0):
            samples.append(
                {
                    "patient_id": str(patient.patient_id),
                    "epoch_index": int(index),
                    "features": features[index].tolist(),
                    "label": int(labels[index]),
                    "apnea_severity": severity,
                    "obesity": obesity,
                }
            )
        return samples

    def _read_recording(self, patient_id: str, event: Any) -> pd.DataFrame | None:
        """Loads the selected channels and stages of one 64 Hz recording.

        Args:
            patient_id: used in warning messages.
            event: the patient's ``dreamt_sleep`` event.

        Returns:
            A DataFrame with ``channels`` and ``Sleep_Stage`` columns and at
            least one full epoch of rows, or ``None`` (after logging a
            warning) if the file is missing, unreadable, empty or too short.
        """
        path = event.attr_dict.get("file_64hz")
        if path is None or (isinstance(path, float) and math.isnan(path)):
            logger.warning("Patient %s has no 64 Hz file", patient_id)
            return None
        path = str(path)
        if not path.strip() or not os.path.isfile(path):
            logger.warning("Patient %s: 64 Hz file not found: %s", patient_id, path)
            return None
        try:
            recording = pd.read_csv(path, usecols=self.channels + ["Sleep_Stage"])
        except (OSError, pd.errors.EmptyDataError, ValueError) as error:
            # ValueError includes pandas' ParserError and missing columns.
            logger.warning("Patient %s: cannot read %s: %s", patient_id, path, error)
            return None
        # A stray non-numeric value becomes NaN instead of failing the cohort.
        recording[self.channels] = recording[self.channels].apply(
            pd.to_numeric, errors="coerce"
        )
        if len(recording) < self.EPOCH_LENGTH:
            logger.warning(
                "Patient %s: %d rows is shorter than one %d-row epoch",
                patient_id,
                len(recording),
                self.EPOCH_LENGTH,
            )
            return None
        return recording

    def _epoch_features(self, signals: np.ndarray, n_epochs: int) -> np.ndarray:
        """Computes mean, std, min and max per epoch and channel.

        Args:
            signals: array of shape ``(n_epochs * EPOCH_LENGTH, n_channels)``.
            n_epochs: number of full epochs.

        Returns:
            float32 array of shape ``(n_epochs, n_channels * 4)``, ordered
            ``[ch1_mean, ch1_std, ch1_min, ch1_max, ch2_mean, ...]``. Infinite
            and out-of-float32-range readings are ignored like NaN, and a
            statistic with no usable reading is 0.
        """
        limit = np.finfo(np.float32).max
        usable = np.isfinite(signals) & (np.abs(signals) <= limit)
        signals = np.where(usable, signals, np.nan)
        epochs = signals.reshape(n_epochs, self.EPOCH_LENGTH, len(self.channels))
        with warnings.catch_warnings():
            # All-NaN epochs warn; their statistics are set to 0 below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            stats = np.stack(
                [
                    np.nanmean(epochs, axis=1),
                    np.nanstd(epochs, axis=1),
                    np.nanmin(epochs, axis=1),
                    np.nanmax(epochs, axis=1),
                ],
                axis=-1,
            )
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return stats.reshape(n_epochs, -1).astype(np.float32)

    @staticmethod
    def _clean_stages(stages: np.ndarray) -> np.ndarray:
        """Returns ``Sleep_Stage`` values as stripped strings (NaN becomes "").

        Args:
            stages: raw ``Sleep_Stage`` values.

        Returns:
            object array of the same length.
        """
        series = pd.Series(stages, dtype="object").fillna("").astype(str)
        return series.str.strip().to_numpy()

    def _grid_segments(self, stages: np.ndarray) -> list[tuple[int, int]]:
        """Finds the PSG's 30-second scoring grid in a recording.

        PSG stages change only on scored-epoch boundaries, so stage changes
        reveal the grid: its offset is a change's row index modulo
        :attr:`EPOCH_LENGTH`. A change counts as an anchor only when the next
        change has the same offset, which ignores isolated label glitches.
        The first anchor sets the grid. A later anchor with a different
        offset (in DREAMT, after a gap that is not a whole number of epochs)
        ends the current segment at its last whole epoch and starts a new
        segment at that anchor.

        Args:
            stages: cleaned ``Sleep_Stage`` values of the whole recording.

        Returns:
            ``(first row, number of epochs)`` for each segment with at least
            one whole epoch, in order. A recording whose stage never changes
            is one segment starting at row 0.
        """
        length = self.EPOCH_LENGTH
        changes = np.flatnonzero(stages[1:] != stages[:-1]) + 1
        offsets = changes % length
        confirmed = np.flatnonzero(offsets[:-1] == offsets[1:])
        anchors = changes[confirmed] if len(confirmed) else changes[:1]
        starts = [int(anchors[0] % length)] if len(anchors) else [0]
        ends = []
        for anchor in anchors[1:]:
            current = starts[-1]
            if (anchor - current) % length:
                ends.append(current + (anchor - current) // length * length)
                starts.append(int(anchor))
        last = starts[-1]
        ends.append(last + (len(stages) - last) // length * length)
        return [
            (start, (end - start) // length)
            for start, end in zip(starts, ends)
            if end > start
        ]

    def _epoch_labels(self, stages: np.ndarray, n_epochs: int) -> np.ndarray:
        """Gives each epoch the most common wake/sleep/unscored class of its rows.

        Each row maps to 1 (``W``), 0 (``N1``/``N2``/``N3``/``R``) or -1
        (``P``, ``Missing`` or any other value); the epoch takes the most
        common class (a plurality, e.g. 40% wake, 35% sleep and 25% unscored
        gives wake). A tie for first gives -1, so mixed epochs are never
        forced into wake or sleep.

        Args:
            stages: cleaned ``Sleep_Stage`` values, length
                ``n_epochs * EPOCH_LENGTH``.
            n_epochs: number of full epochs.

        Returns:
            int array of length ``n_epochs``: 1 (wake), 0 (sleep) or -1
            (not a target).
        """
        row_labels = (
            pd.Series(stages, dtype="object")
            .map(self.STAGE_TO_LABEL)
            .fillna(-1)
            .to_numpy(dtype=np.int64)
            .reshape(n_epochs, self.EPOCH_LENGTH)
        )
        classes = np.array([-1, 0, 1], dtype=np.int64)
        counts = np.stack([(row_labels == c).sum(axis=1) for c in classes], axis=1)
        labels = classes[counts.argmax(axis=1)]
        tied = (counts == counts.max(axis=1, keepdims=True)).sum(axis=1) > 1
        labels[tied] = -1
        return labels

    def _add_context(self, features: np.ndarray) -> np.ndarray:
        """Concatenates each epoch with its ``context_window`` neighbours.

        Args:
            features: array of shape ``(n_epochs, n_features)``.

        Returns:
            array of shape ``(n_epochs, n_features * (2k + 1))``, ordered from
            epoch ``i - k`` to ``i + k``; past the edges the first or last
            epoch is repeated.
        """
        k = self.context_window
        if k == 0:
            return features
        n_epochs = features.shape[0]
        padded = np.pad(features, ((k, k), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1, axis=0)
        # windows: (n_epochs, n_features, 2k + 1) -> (n_epochs, 2k + 1, n_features)
        return windows.transpose(0, 2, 1).reshape(n_epochs, -1)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Parses a number that may arrive as a string, or returns ``None``.

        Args:
            value: e.g. ``31.5``, ``"31.5"``, ``None`` or ``"n/a"``.

        Returns:
            The value as a finite float, or ``None`` if it is missing or not
            numeric.
        """
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def apnea_severity_from_ahi(ahi: Any) -> str:
        """Maps an apnea-hypopnea index (events per hour) to a severity group.

        Uses the standard clinical cut-offs, as in the paper's apnea severity
        random effect.

        Args:
            ahi: the AHI as a number or numeric string; anything else counts as
                missing.

        Returns:
            ``"none"`` (< 5), ``"mild"`` (5 to < 15), ``"moderate"`` (15 to
            < 30), ``"severe"`` (>= 30) or ``"unknown"``.

        Examples:
            >>> SleepWakeDREAMT.apnea_severity_from_ahi(12)
            'mild'
            >>> SleepWakeDREAMT.apnea_severity_from_ahi("30.0")
            'severe'
            >>> SleepWakeDREAMT.apnea_severity_from_ahi(None)
            'unknown'
        """
        value = SleepWakeDREAMT._to_float(ahi)
        if value is None:
            return "unknown"
        if value < 5:
            return "none"
        if value < 15:
            return "mild"
        if value < 30:
            return "moderate"
        return "severe"

    @staticmethod
    def obesity_from_bmi(bmi: Any) -> int:
        """Flags obesity (BMI >= 30 kg/m^2), as the DREAMT paper defines it.

        The paper (Section 2.3) counts 68 participants as "obese or severely
        obese (BMI >= 30 kg/m^2)"; this cut-off reproduces that count.

        Args:
            bmi: the BMI as a number or numeric string; anything else counts as
                missing.

        Returns:
            1 if the BMI is at least 30, 0 if it is below 30, and -1 if it is
            missing, so missing values are not counted as non-obese.

        Examples:
            >>> SleepWakeDREAMT.obesity_from_bmi("31.0")
            1
            >>> SleepWakeDREAMT.obesity_from_bmi(23)
            0
            >>> SleepWakeDREAMT.obesity_from_bmi(None)
            -1
        """
        value = SleepWakeDREAMT._to_float(bmi)
        if value is None:
            return -1
        return int(value >= 30)
