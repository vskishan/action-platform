"""
Survival Analysis Engine

Fits a Cox Proportional-Hazards (CoxPH) model to clinical trial data
(demographics, disease progression, mortality, assessments, and labs)
and exposes a prediction API for downstream consumers.

Column glossary (source datasets use abbreviated CDISC-style names):
    TRIAL      - trial identifier
    TRIALS_D   - subject identifier within a trial
    DYPROG     - day of disease progression
    DYDTH      - day of death
    VISDY      - visit day (assessments)
    DYLAB      - lab-test day
    AGEGRP     - age-group bucket (e.g. '60-64')
    GLEASON    - Gleason score (prostate cancer grading)
    PSA        - prostate-specific antigen level
    BONEMET    - bone-metastasis indicator
    POSNODE    - positive lymph-node indicator
    BILI       - bilirubin lab value
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

logger = logging.getLogger(__name__)

# Configuration

# Column names used as the composite subject key across all source files.
SUBJECT_KEY: list[str] = ["TRIAL", "TRIALS_D"]

# Default directory and filename for persisted model artefacts.
# Resolve relative to *this* file so the path is correct regardless of cwd.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
_MODEL_FILENAME = "coxph_survival_model.pkl"


@dataclass(frozen=True)
class DataPaths:
    """Centralised paths to the CSV data files.

    Defaults point to ``data/control_arm/`` relative to the working directory.
    """

    demographics: Path = Path("data/control_arm/demographics.csv")
    progression: Path = Path("data/control_arm/progression.csv")
    mortality: Path = Path("data/control_arm/mortality.csv")
    assessment: Path = Path("data/control_arm/assessment.csv")
    lab_tests: Path = Path("data/control_arm/lab_test_details.csv")

# Age-group helpers

_YOUNG_GROUPS = frozenset({"35-39", "40-44", "45-49", "50-54", "55-59"})
_MIDDLE_GROUPS = frozenset({"60-64", "65-69"})


def collapse_age_group(age_group: str) -> str:
    """Collapse granular 5-year age buckets into three broad categories.

    Mapping:
        35-59 → '<60'
        60-69 → '60-69'
        70+   → '70+'

    Parameters
    ----------
    age_group : str
        Original 5-year age-group label (e.g. ``'60-64'``).

    Returns
    -------
    str
        Collapsed category.
    """
    if age_group in _YOUNG_GROUPS:
        return "<60"
    if age_group in _MIDDLE_GROUPS:
        return "60-69"
    return "70+"


class SurvivalAnalysisEngine:
    """Cox Proportional-Hazards survival analysis engine.

    Typical usage:

        engine = SurvivalAnalysisEngine()
        engine.run()                       # loads data, engineers features, fits model
        engine.print_summary()             # display model summary & C-index
        predictions = engine.predict(df)   # score new observations

    Parameters
    ----------
    data_paths : DataPaths, optional
        Override default CSV locations.
    penalizer : float
        L2 penalizer passed to :class:`lifelines.CoxPHFitter` (default 0.0).
    """

    # Columns selected for the final modelling dataset.
    _MODEL_FEATURES: list[str] = [
        "AGEGRP",
        "GLEASON",
        "log_psa",
        "BONEMET",
        "POSNODE",
        "event_time",
        "event",
    ]

    # Categorical columns to one-hot encode (drop_first to avoid collinearity).
    _CATEGORICAL_COLS: list[str] = ["AGE", "BONEMET", "POSNODE"]

    def __init__(
        self,
        data_paths: Optional[DataPaths] = None,
        penalizer: float = 0.0,
        model_dir: Optional[Path] = None,
    ) -> None:
        self._paths = data_paths or DataPaths()
        self._penalizer = penalizer
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._model_path = self._model_dir / _MODEL_FILENAME

        self._cph: Optional[CoxPHFitter] = None
        self._model_df: Optional[pd.DataFrame] = None

    # Public interface

    def run(self) -> "SurvivalAnalysisEngine":
        """Load a saved model if available, otherwise train from scratch.

        The trained model is persisted to ``model_dir`` so that
        subsequent calls skip the expensive training pipeline.

        Returns ``self`` so callers can chain, e.g.
        ``engine = SurvivalAnalysisEngine().run()``.
        """
        # Attempt to load a previously saved model.
        if self._load_model():
            logger.info("Loaded saved model from %s", self._model_path)
            return self

        # No saved model found — run the full training pipeline.
        logger.info("No saved model found. Starting training pipeline.")
        self._train()
        self._save_model()

        return self

    def retrain(self) -> "SurvivalAnalysisEngine":
        """Force re-training regardless of whether a saved model exists.

        Useful when the underlying data has changed.

        Returns ``self`` for chaining.
        """
        logger.info("Forcing re-training of CoxPH model.")
        self._train()
        self._save_model()
        return self

    def _train(self) -> None:
        """Run the full data pipeline: load → merge → engineer → fit."""
        demographics = self._load_demographics()
        demographics = self._merge_progression(demographics)
        demographics = self._merge_mortality(demographics)
        demographics = self._merge_baseline_assessments(demographics)
        demographics = self._merge_baseline_labs(demographics)

        model_df = self._engineer_features(demographics)
        self._fit(model_df)

    def print_summary(self) -> None:
        """Print the fitted CoxPH model summary and concordance index."""
        self._ensure_fitted()
        self._cph.print_summary()
        logger.info("C-index: %.4f", self._cph.concordance_index_)

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return partial-hazard predictions for new observations.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Must contain the same dummy-encoded columns the model was
            trained on (excluding ``event_time`` and ``event``).

        Returns
        -------
        pd.DataFrame
            Partial hazard predictions.
        """
        self._ensure_fitted()
        return self._cph.predict_partial_hazard(dataframe)

    @property
    def concordance_index(self) -> float:
        """Return the C-index of the fitted model."""
        self._ensure_fitted()
        return self._cph.concordance_index_

    @property
    def fitted_model(self) -> CoxPHFitter:
        """Return the underlying fitted :class:`CoxPHFitter` instance."""
        self._ensure_fitted()
        return self._cph

    @property
    def training_data(self) -> pd.DataFrame:
        """Return a *copy* of the processed training dataframe."""
        if self._model_df is None:
            raise RuntimeError("Model has not been trained yet. Call run() first.")
        return self._model_df.copy()

    # Data loading

    def _load_demographics(self) -> pd.DataFrame:
        """Load the demographics (control arm) CSV."""
        logger.info("Loading demographics from %s", self._paths.demographics)
        return pd.read_csv(self._paths.demographics)

    # Data merging

    def _merge_progression(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Merge earliest disease-progression day onto the base demographics data.

        Only rows with a non-null ``DYPROG`` (progression day) are kept
        from the source; the *earliest* event per subject is selected.
        """
        progression = pd.read_csv(self._paths.progression)

        earliest_progression = (
            progression[SUBJECT_KEY + ["DYPROG"]]
            .dropna(subset=["DYPROG"])
            .groupby(SUBJECT_KEY, as_index=False)
            .min()
        )

        logger.info(
            "Merged %d progression events onto %d subjects",
            len(earliest_progression),
            len(demographics),
        )
        return demographics.merge(earliest_progression, on=SUBJECT_KEY, how="left")

    def _merge_mortality(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Merge earliest death day onto the base demographics data."""
        mortality = pd.read_csv(self._paths.mortality)

        earliest_death = (
            mortality[mortality["DYDTH"].notna()]
            .groupby(SUBJECT_KEY, as_index=False)["DYDTH"]
            .min()
        )

        logger.info(
            "Merged %d mortality events onto %d subjects",
            len(earliest_death),
            len(demographics),
        )
        return demographics.merge(earliest_death, on=SUBJECT_KEY, how="left")

    def _merge_baseline_assessments(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Merge baseline (first-visit) assessment observations.

        Extracts ``BONEMET`` and ``POSNODE`` from the earliest visit.
        """
        assessments = pd.read_csv(self._paths.assessment).sort_values("VISDY")

        baseline = (
            assessments
            .groupby(SUBJECT_KEY, as_index=False)
            .first()[SUBJECT_KEY + ["BONEMET", "POSNODE"]]
        )

        logger.info("Merged baseline assessments for %d subjects", len(baseline))
        return demographics.merge(baseline, on=SUBJECT_KEY, how="left")

    def _merge_baseline_labs(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Merge baseline (first-day) lab values: PSA and bilirubin."""
        labs = pd.read_csv(self._paths.lab_tests).sort_values("DYLAB")

        baseline = (
            labs
            .groupby(SUBJECT_KEY, as_index=False)
            .first()[SUBJECT_KEY + ["PSA", "BILI"]]
        )

        logger.info("Merged baseline lab values for %d subjects", len(baseline))
        return demographics.merge(baseline, on=SUBJECT_KEY, how="left")

    # Feature engineering

    def _engineer_features(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """Derive modelling features from the merged cohort.

        Steps:
        1. Compute ``log_psa`` = log(PSA + 1) to reduce skew.
        2. Compute ``event_time`` as the *earlier* of progression or death.
        3. Compute a binary ``event`` indicator (1 = event observed).
        4. Censor subjects with no event at the maximum observed follow-up.
        5. Collapse age groups and one-hot encode categoricals.
        6. Drop rows with any remaining missing values.
        """
        df = demographics.copy()

        # Log-transform PSA (add 1 to handle zeros).
        df["log_psa"] = np.log(df["PSA"] + 1)

        # Composite event time: whichever came first — progression or death.
        df["event_time"] = df[["DYPROG", "DYDTH"]].min(axis=1)

        # Binary indicator: 1 if any event was observed, 0 if censored.
        df["event"] = df["event_time"].notna().astype(int)

        # Censoring: subjects with no event are assigned the latest observed
        # follow-up time in the dataset (administrative censoring).
        max_followup = df[["DYPROG", "DYDTH"]].max().max()
        df["event_time"] = df["event_time"].fillna(max_followup)
        logger.info("Max follow-up (censoring time): %s days", max_followup)

        # Select relevant model columns.
        model_df = df[self._MODEL_FEATURES].copy()

        # Collapse granular age groups into broader categories.
        model_df["AGE"] = model_df["AGEGRP"].apply(collapse_age_group)
        model_df = model_df.drop(columns=["AGEGRP"])

        # Report and drop missing values.
        missing_counts = model_df.isna().sum()
        if missing_counts.any():
            logger.warning("Dropping rows with missing values:\n%s", missing_counts)
        model_df = model_df.dropna()

        # One-hot encode categorical variables (drop first level to avoid
        # perfect multicollinearity in the CoxPH design matrix).
        model_df = pd.get_dummies(
            model_df,
            columns=self._CATEGORICAL_COLS,
            drop_first=True,
        )

        logger.info(
            "Final modelling dataset: %d rows × %d columns",
            *model_df.shape,
        )
        self._model_df = model_df
        return model_df

    # Model fitting

    def _fit(self, model_df: pd.DataFrame) -> None:
        """Fit a Cox Proportional-Hazards model.

        Parameters
        ----------
        model_df : pd.DataFrame
            Processed dataframe with ``event_time`` and ``event`` columns
            plus covariate dummies.
        """
        self._cph = CoxPHFitter(penalizer=self._penalizer)
        self._cph.fit(
            model_df,
            duration_col="event_time",
            event_col="event",
        )
        logger.info(
            "CoxPH model fitted — C-index: %.4f",
            self._cph.concordance_index_,
        )

    # Model persistence

    def _save_model(self) -> None:
        """Persist the fitted model and training data to disk."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        artefact = {
            "cph": self._cph,
            "model_df": self._model_df,
            "penalizer": self._penalizer,
        }

        with open(self._model_path, "wb") as fh:
            pickle.dump(artefact, fh)

        logger.info("Model saved to %s", self._model_path)

    def _load_model(self) -> bool:
        """Load a previously saved model from disk.

        Returns
        -------
        bool
            ``True`` if a saved model was loaded successfully,
            ``False`` otherwise.
        """
        if not self._model_path.exists():
            return False

        try:
            with open(self._model_path, "rb") as fh:
                artefact = pickle.load(fh)

            self._cph = artefact["cph"]
            self._model_df = artefact["model_df"]
            logger.info(
                "Restored model (C-index: %.4f) from %s",
                self._cph.concordance_index_,
                self._model_path,
            )
            return True

        except (pickle.UnpicklingError, KeyError, EOFError) as exc:
            logger.warning(
                "Failed to load saved model from %s: %s. "
                "Will retrain from scratch.",
                self._model_path,
                exc,
            )
            return False

    # Guards

    def _ensure_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if self._cph is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call run() first."
            )


# Standalone execution

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    engine = SurvivalAnalysisEngine()
    engine.run()
    engine.print_summary()