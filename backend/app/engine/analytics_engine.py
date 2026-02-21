"""
Analytics Engine

Provides descriptive analytics and statistical summaries over the
AI based control arm data (demographics, progression, mortality, assessments and labs).

Handles intents such as:
    - progression_stats              : disease-progression rates and timelines
    - mortality_stats                : mortality rates, causes of death breakdown
    - lab_summary                    : baseline lab-value distributions (PSA, bilirubin, etc.)
    - assessment_summary             : bone-metastasis and positive-node prevalence
    - gleason_distribution           : Gleason score distribution across the cohort
    - adverse_events_by_demographics : adverse-event rates and severity broken down by
                                       age group and race
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.app.engine.base_engine import BaseEngine

logger = logging.getLogger(__name__)

# Re-use shared constants from the survival engine.
SUBJECT_KEY: list[str] = ["TRIAL", "TRIALS_D"]

# CTCAE intensity codes → human-readable severity labels.
_SEVERITY_MAP = {1: "mild", 2: "moderate", 3: "severe"}

# Resolve data directory relative to this file.
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "control_arm"


class AnalyticsEngine(BaseEngine):
    """Descriptive-analytics engine over AI based control arm data.

    Typical usage::

        engine = AnalyticsEngine().run()
        result = engine.query("gleason_distribution", {})
    """

    _INTENTS: list[str] = [
        "progression_stats",
        "mortality_stats",
        "lab_summary",
        "assessment_summary",
        "gleason_distribution",
        "adverse_events_by_demographics",
    ]

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self._data_dir = data_dir or _DATA_DIR

        # Dataframes loaded on run().
        self._demographics: Optional[pd.DataFrame] = None
        self._progression: Optional[pd.DataFrame] = None
        self._mortality: Optional[pd.DataFrame] = None
        self._assessment: Optional[pd.DataFrame] = None
        self._labs: Optional[pd.DataFrame] = None
        self._adverse_events: Optional[pd.DataFrame] = None

    # BaseEngine interface

    def run(self) -> "AnalyticsEngine":
        """Load all CSV files into memory."""
        logger.info("Loading analytics data from %s", self._data_dir)

        self._demographics = pd.read_csv(self._data_dir / "demographics.csv")
        self._progression = pd.read_csv(self._data_dir / "progression.csv")
        self._mortality = pd.read_csv(self._data_dir / "mortality.csv")
        self._assessment = pd.read_csv(self._data_dir / "assessment.csv")
        self._labs = pd.read_csv(self._data_dir / "lab_test_details.csv")
        self._adverse_events = pd.read_csv(self._data_dir / "adverse_events.csv")

        logger.info(
            "Analytics data loaded — %d subjects in demographics",
            len(self._demographics),
        )
        return self

    def query(self, intent: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Route to the appropriate analytics handler."""
        self._ensure_loaded()

        handler_map = {
            "progression_stats": self._progression_stats,
            "mortality_stats": self._mortality_stats,
            "lab_summary": self._lab_summary,
            "assessment_summary": self._assessment_summary,
            "gleason_distribution": self._gleason_distribution,
            "adverse_events_by_demographics": self._adverse_events_by_demographics,
        }

        handler = handler_map.get(intent)
        if handler is None:
            return {
                "error": f"Unknown analytics intent: '{intent}'",
                "supported_intents": self._INTENTS,
            }

        return handler(parameters)

    @property
    def capabilities(self) -> list[str]:
        """Return the list of intent strings this engine can handle."""
        return list(self._INTENTS)

    def _progression_stats(self, params: dict[str, Any]) -> dict[str, Any]:
        """Disease-progression rates and timelines."""
        prog = self._progression.dropna(subset=["DYPROG"])
        total_subjects = len(self._demographics)

        subjects_with_progression = prog[SUBJECT_KEY].drop_duplicates()
        progression_count = len(subjects_with_progression)

        # Earliest progression per subject.
        earliest = prog.groupby(SUBJECT_KEY, as_index=False)["DYPROG"].min()

        return {
            "intent": "progression_stats",
            "total_subjects": total_subjects,
            "subjects_with_progression": progression_count,
            "progression_rate_pct": round(
                100 * progression_count / total_subjects, 2
            ),
            "median_days_to_progression": round(float(earliest["DYPROG"].median()), 1),
            "mean_days_to_progression": round(float(earliest["DYPROG"].mean()), 1),
            "min_days_to_progression": round(float(earliest["DYPROG"].min()), 1),
            "max_days_to_progression": round(float(earliest["DYPROG"].max()), 1),
        }

    def _mortality_stats(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mortality rates and cause-of-death breakdown."""
        mort = self._mortality.dropna(subset=["DYDTH"])
        total_subjects = len(self._demographics)

        subjects_with_death = mort[SUBJECT_KEY].drop_duplicates()
        death_count = len(subjects_with_death)

        # Earliest death per subject.
        earliest = mort.groupby(SUBJECT_KEY, as_index=False)["DYDTH"].min()

        # Cause of death breakdown (first record per subject).
        cause_breakdown = (
            mort.sort_values("DYDTH")
            .groupby(SUBJECT_KEY, as_index=False)
            .first()["CAUSE"]
            .value_counts()
            .to_dict()
        )

        return {
            "intent": "mortality_stats",
            "total_subjects": total_subjects,
            "subjects_deceased": death_count,
            "mortality_rate_pct": round(100 * death_count / total_subjects, 2),
            "median_days_to_death": round(float(earliest["DYDTH"].median()), 1),
            "mean_days_to_death": round(float(earliest["DYDTH"].mean()), 1),
            "cause_of_death_breakdown": cause_breakdown,
        }

    def _lab_summary(self, params: dict[str, Any]) -> dict[str, Any]:
        """Baseline lab-value distributions (PSA, bilirubin, AST, ALT)."""
        labs = self._labs.sort_values("DYLAB")

        # Take the earliest lab record per subject.
        baseline = labs.groupby(SUBJECT_KEY, as_index=False).first()

        lab_cols = ["PSA", "BILI", "AST", "ALT"]
        summaries = {}

        for col in lab_cols:
            series = baseline[col].dropna()
            if len(series) == 0:
                continue
            summaries[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
            }

        return {
            "intent": "lab_summary",
            "baseline_subjects": int(len(baseline)),
            "lab_statistics": summaries,
        }

    def _assessment_summary(self, params: dict[str, Any]) -> dict[str, Any]:
        """Bone-metastasis and positive-node prevalence at baseline."""
        assessments = self._assessment.sort_values("VISDY")

        # Take the baseline (first visit) per subject.
        baseline = (
            assessments.groupby(SUBJECT_KEY, as_index=False)
            .first()[SUBJECT_KEY + ["BONEMET", "POSNODE"]]
        )

        bonemet_counts = baseline["BONEMET"].value_counts().to_dict()
        posnode_counts = baseline["POSNODE"].value_counts().to_dict()

        bonemet_total = baseline["BONEMET"].notna().sum()
        posnode_total = baseline["POSNODE"].notna().sum()

        return {
            "intent": "assessment_summary",
            "baseline_subjects": int(len(baseline)),
            "bone_metastasis": {
                "distribution": bonemet_counts,
                "available_records": int(bonemet_total),
            },
            "positive_nodes": {
                "distribution": posnode_counts,
                "available_records": int(posnode_total),
            },
        }

    def _gleason_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        """Gleason score distribution across the cohort."""
        df = self._demographics

        distribution = (
            df["GLEASON"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        # Convert keys to strings for JSON serialisability.
        distribution = {str(k): v for k, v in distribution.items()}

        return {
            "intent": "gleason_distribution",
            "total_subjects": int(len(df)),
            "distribution": distribution,
            "mean": round(float(df["GLEASON"].mean()), 2),
            "median": int(df["GLEASON"].median()),
            "min": int(df["GLEASON"].min()),
            "max": int(df["GLEASON"].max()),
        }

    def _adverse_events_by_demographics(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Adverse-event rates and severity stratified by age group and race."""

        # Merge adverse events with demographics to get age group and race.
        ae_demo = self._adverse_events.merge(
            self._demographics[SUBJECT_KEY + ["AGEGRP", "RACE"]],
            on=SUBJECT_KEY,
            how="inner",
        )

        total_subjects = len(self._demographics)
        subjects_with_ae = ae_demo[SUBJECT_KEY].drop_duplicates()
        ae_subject_count = len(subjects_with_ae)

        # Overall summary
        overall = {
            "total_ae_records": int(len(ae_demo)),
            "subjects_with_any_ae": ae_subject_count,
            "ae_rate_pct": round(100 * ae_subject_count / total_subjects, 2),
            "severity_distribution": (
                ae_demo["INTEN"]
                .map(_SEVERITY_MAP)
                .value_counts()
                .to_dict()
            ),
            "hospitalisation_count": int((ae_demo["HOSP"] == "Y").sum()),
            "life_threatening_count": int((ae_demo["LTHREAT"] == "Y").sum()),
        }

        # Breakdown by age group
        by_age: dict[str, Any] = {}
        for age_grp, grp in ae_demo.groupby("AGEGRP"):
            total_in_age = len(
                self._demographics[self._demographics["AGEGRP"] == age_grp]
            )
            by_age[str(age_grp)] = self._compute_group_metrics(
                grp, total_in_age,
            )

        # Breakdown by race
        race_label_map = {
            "W": "White",
            "B": "Black",
            "H": "Hispanic",
            "O": "Other",
            "X": "Unknown",
        }
        by_race: dict[str, Any] = {}
        for race_code, grp in ae_demo.groupby("RACE"):
            total_in_race = len(
                self._demographics[self._demographics["RACE"] == race_code]
            )
            label = race_label_map.get(str(race_code), str(race_code))
            by_race[label] = self._compute_group_metrics(
                grp, total_in_race,
            )

        # Median onset day by age group
        onset_by_age = (
            ae_demo.groupby("AGEGRP")["DYSTRT"]
            .median()
            .round(1)
            .to_dict()
        )
        onset_by_age = {str(k): float(v) for k, v in onset_by_age.items()}

        return {
            "intent": "adverse_events_by_demographics",
            "total_subjects": total_subjects,
            "overall": overall,
            "by_age_group": by_age,
            "by_race": by_race,
            "median_onset_day_by_age_group": onset_by_age,
        }

    # Guards

    def _ensure_loaded(self) -> None:
        """Raise if data has not been loaded yet."""
        if self._demographics is None:
            raise RuntimeError(
                "Analytics data has not been loaded. Call run() first."
            )

    # Reusable helpers

    @staticmethod
    def _compute_group_metrics(
        grp: pd.DataFrame,
        total_in_group: int,
    ) -> dict[str, Any]:
        """Compute AE metrics for a single demographic sub-group.

        This is shared between the age-group and race breakdowns to
        avoid duplicating the same rate / count calculations.
        """
        group_subjects = grp[SUBJECT_KEY].drop_duplicates()
        n_subjects = len(group_subjects)
        n_records = len(grp)

        severity_counts = (
            grp["INTEN"].map(_SEVERITY_MAP).value_counts().to_dict()
        )
        top_aes = grp["KEYTEXT"].value_counts().head(5).to_dict()

        return {
            "total_subjects_in_group": int(total_in_group),
            "subjects_with_ae": int(n_subjects),
            "ae_rate_pct": (
                round(100 * n_subjects / total_in_group, 2)
                if total_in_group > 0 else 0.0
            ),
            "total_ae_records": int(n_records),
            "mean_ae_per_subject": (
                round(n_records / n_subjects, 2) if n_subjects > 0 else 0.0
            ),
            "severity_distribution": severity_counts,
            "top_adverse_events": top_aes,
            "hospitalisation_rate_pct": (
                round(100 * (grp["HOSP"] == "Y").sum() / n_records, 2)
                if n_records > 0 else 0.0
            ),
            "life_threatening_rate_pct": (
                round(100 * (grp["LTHREAT"] == "Y").sum() / n_records, 2)
                if n_records > 0 else 0.0
            ),
        }


# Standalone smoke test.
if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    engine = AnalyticsEngine().run()

    for intent in engine.capabilities:
        result = engine.query(intent, {})
        print(f"\n{'=' * 60}")
        print(f"Intent: {intent}")
        print(f"{'=' * 60}")
        print(json.dumps(result, indent=2, default=str))
