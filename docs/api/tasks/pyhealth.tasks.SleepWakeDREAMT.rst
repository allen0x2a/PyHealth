pyhealth.tasks.SleepWakeDREAMT
====================================

Sleep/wake detection on DREAMT (Wang et al., CHIL 2024,
https://proceedings.mlr.press/v248/wang24a.html).

For each participant the task reads the 64 Hz Empatica E4 recording, keeps
the selected channels (ACC_X, ACC_Y, ACC_Z, HR, TEMP, EDA and BVP by
default), and cuts it into 30-second epochs aligned to the PSG scoring
grid (re-aligned where the grid shifts after a gap). Each epoch's features are the mean, standard deviation, minimum and
maximum of every channel, optionally concatenated with those of
``context_window`` neighbouring epochs on each side (the first and last epoch
are repeated past the edges of the recording).

Labels are 1 for wake (W) and 0 for sleep (N1, N2, N3 and REM), by the most
common class among the epoch's rows. Epochs where P (preparation) or Missing
is most common, or with a tie, are not samples. Recordings that are missing, empty
or shorter than one epoch are skipped with a warning. Each sample also
carries the participant's apnea severity (from the AHI) and obesity (BMI of
30 or more; -1 if missing), the groups the paper uses as random effects.

Task parameters:

- context_window: int, default=0, neighbouring epochs on each side
- channels: list or tuple of signal columns, default
  ``SleepWakeDREAMT.DEFAULT_CHANNELS``; ``"IBI"`` is also accepted

.. autoclass:: pyhealth.tasks.SleepWakeDREAMT
    :members:
    :undoc-members:
    :show-inheritance:
