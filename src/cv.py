"""Temporal validation: train on seasons < val_season, validate on val_season.
Mirrors the test setup (train<=2024 -> predict 2025)."""
import numpy as np

FOLDS = [2024, 2023]  # primary fold first


def temporal_folds(seasons, folds=FOLDS, min_train_season=2019):
    seasons = np.asarray(seasons)
    for vs in folds:
        tr_idx = np.where((seasons < vs) & (seasons >= min_train_season))[0]
        va_idx = np.where(seasons == vs)[0]
        yield vs, tr_idx, va_idx
