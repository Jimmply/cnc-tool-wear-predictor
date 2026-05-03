"""
CNC Tool Wear Synthetic Data Generator
=======================================
Simulates multi-sensor readings over a tool's cutting life for a fleet
of milling machine spindles.

Physics modeled:
  - Wear index follows a 3-phase curve: break-in → steady-state → accelerated
  - Sensors degrade monotonically with wear, with per-tool variability
  - Micro-chipping events inject kurtosis spikes in the worn/critical phase

Output columns:
  tool_id, cut_number, wear_index, vibration_rms_mm_s, vibration_kurtosis,
  spindle_current_a, acoustic_emission_db, cutting_force_n,
  surface_roughness_um, wear_state, rul_cycles
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEAR_STATE_COLORS = {"Fresh": "#2ecc71", "Worn": "#f39c12", "Critical": "#e74c3c"}
FEATURE_COLS = [
    "vibration_rms_mm_s",
    "vibration_kurtosis",
    "spindle_current_a",
    "acoustic_emission_db",
    "cutting_force_n",
    "surface_roughness_um",
]


@dataclass
class ToolConfig:
    """Lifecycle parameters shared across the simulated tool fleet."""
    max_cuts: int = 500
    break_in_end: float = 0.05    # Fraction of life where break-in ends
    accel_start: float = 0.72     # Fraction where accelerated wear begins
    fresh_threshold: float = 0.30
    critical_threshold: float = 0.70


class ToolWearGenerator:
    """
    Generates synthetic wear telemetry for a fleet of milling tools.

    Parameters
    ----------
    n_tools : int
        Number of tools to simulate.
    tool_config : ToolConfig, optional
        Lifecycle parameters.  Defaults to ToolConfig().
    random_seed : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_tools: int = 20,
        tool_config: Optional[ToolConfig] = None,
        random_seed: int = 42,
    ) -> None:
        self.n_tools = n_tools
        self.config = tool_config or ToolConfig()
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate and concatenate wear data for all tools."""
        frames: List[pd.DataFrame] = [
            self._generate_tool(i) for i in range(self.n_tools)
        ]
        df = pd.concat(frames, ignore_index=True)
        logger.info("Generated %d records for %d tools.", len(df), self.n_tools)
        return df

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _wear_curve(self, n: int) -> np.ndarray:
        """3-phase wear index (0→1): break-in / steady-state / accelerated."""
        cfg = self.config
        t = np.linspace(0, 1, n)
        p1, p2 = cfg.break_in_end, cfg.accel_start
        wear = np.empty(n)

        for i, ti in enumerate(t):
            if ti <= p1:
                wear[i] = 0.15 * (ti / p1) ** 0.5
            elif ti <= p2:
                frac = (ti - p1) / (p2 - p1)
                wear[i] = 0.15 + 0.45 * frac
            else:
                frac = (ti - p2) / (1.0 - p2)
                wear[i] = 0.60 + 0.40 * (np.exp(2.5 * frac) - 1) / (np.exp(2.5) - 1)

        return np.clip(wear, 0, 1)

    def _generate_tool(self, tool_id: int) -> pd.DataFrame:
        cfg = self.config

        # Per-tool variability: ±15% life span, small baseline offsets
        max_cuts = int(cfg.max_cuts * self._rng.uniform(0.85, 1.15))
        n = max_cuts
        cuts = np.arange(1, n + 1)
        wear = self._wear_curve(n)
        wear_obs = np.clip(wear + self._rng.normal(0, 0.008, n), 0, 1)

        # --- Sensor degradation models ---------------------------------

        # Vibration RMS (mm/s): 0.4 → 3.6
        vib_rms = (
            0.4 + 3.2 * wear ** 1.4
            + self._rng.normal(0, 0.04, n)
            + self._rng.uniform(-0.04, 0.04)
        ).clip(0.1)

        # Vibration kurtosis: 3.0 → 14 + micro-chip spikes
        vib_kurt = 3.0 + 11.0 * wear ** 2.0 + self._rng.normal(0, 0.25, n)
        n_chips = self._rng.integers(2, 7)
        chip_locs = self._rng.integers(int(n * 0.35), n, size=n_chips)
        vib_kurt[chip_locs] += self._rng.uniform(3, 9, size=n_chips)

        # Spindle current (A): 7.5 → 15.5
        spindle_a = (
            7.5 + 8.0 * wear ** 1.2
            + self._rng.normal(0, 0.18, n)
            + self._rng.uniform(-0.25, 0.25)
        ).clip(5.0)

        # Acoustic emission (dB): 55 → 86
        ae_db = (
            55.0 + 31.0 * wear
            + self._rng.normal(0, 0.7, n)
        ).clip(50.0)

        # Cutting force (N): 180 → 530
        force_n = (
            180.0 + 350.0 * wear ** 1.3
            + self._rng.normal(0, 7.0, n)
            + self._rng.uniform(-8, 8)
        ).clip(100.0)

        # Surface roughness Ra (μm): 0.6 → 4.2
        roughness = (
            0.6 + 3.6 * wear ** 1.6
            + self._rng.normal(0, 0.04, n)
        ).clip(0.2)

        wear_state = np.where(
            wear_obs < cfg.fresh_threshold, "Fresh",
            np.where(wear_obs < cfg.critical_threshold, "Worn", "Critical"),
        )

        return pd.DataFrame({
            "tool_id": f"TOOL-{tool_id:03d}",
            "cut_number": cuts,
            "wear_index": wear_obs.round(4),
            "vibration_rms_mm_s": vib_rms.round(3),
            "vibration_kurtosis": vib_kurt.round(3),
            "spindle_current_a": spindle_a.round(3),
            "acoustic_emission_db": ae_db.round(2),
            "cutting_force_n": force_n.round(1),
            "surface_roughness_um": roughness.round(3),
            "wear_state": wear_state,
            "rul_cycles": (max_cuts - cuts).clip(0),
        })
