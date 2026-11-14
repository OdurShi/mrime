"""Configuration of RIME / MRIME-CD: strategy switches, hyper-parameters, compatibility profiles.

Every field documents where its value comes from: the paper (equation / table number), the
author's MATLAB code (file name), or a choice recorded in ``docs/implementation_notes.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

ThetaMode = Literal["paper", "matlab"]
WeightOrder = Literal["fitness", "distance"]
AbsReference = Literal["best", "roulette"]
AbsGate = Literal["shared", "independent"]
Accounting = Literal["strict", "matlab-as-shipped"]


@dataclass(frozen=True)
class MRIMEConfig:
    """All switches and constants of the RIME family implemented in :mod:`mrime.core`.

    With the three strategy switches off this is the base RIME algorithm (paper Section 2);
    with all three on it is MRIME-CD (Section 3, Algorithm 1). The other combinations are the
    ablation variants of Table 5.

    Strategy switches (Table 5):
        use_gcls: Gaussian-based covariance learning strategy, Eqs. (9)-(13).
        use_abs: average bootstrapping strategy, Eq. (14).
        use_spdm: stochastic covariance learning-based population diversity mechanism,
            Eqs. (15)-(18).

    Base RIME (Section 2, Table 2):
        w: number of steps of the soft-rime step function, Eq. (6); ``w = 5`` in Table 2.
        theta_mode: ``"paper"`` uses ``theta = pi * FEs / (10 * FEsmax)`` exactly as printed in
            Eq. (5); ``"matlab"`` uses ``theta = 10 * pi * FEs / FEsmax`` as coded in every
            ``matlab/*RIME*.m`` file (``cos(pi*it/(FesMax/10))``).

    Dominant group (Section 3.1 gives the mechanism but no numbers; values from
    ``matlab/MRIME_RL_Cov.m``):
        elite_fraction: share of the current population (best by fitness) appended to the
            first-in-first-out archive every generation (``round(N*0.3)``).
        archive_capacity_per_dim: archive capacity is this value times ``dim`` (``25*dim``).
        neighbour_fraction: share of the archive, nearest to the roulette-selected ``X_r``,
            that forms the dominant group ``S`` (``floor(0.5*|archive|)``).

    Gaussian model (Eqs. 9-11):
        weight_offset: ``c`` in ``w_i ~ ln(|S| + c) - ln(i)``. The paper prints ``c = 1``
            (Eq. 10); the MATLAB code uses ``c = 0.5``.
        weight_order: ranking that assigns the weights. ``"fitness"`` (best first) follows the
            paper's statement that better individuals get larger weights; ``"distance"``
            (nearest to ``X_r`` first) is what the MATLAB code does.
        cov_ddof: ``0`` gives the ``1/|S|`` normalisation of Eq. (11); ``1`` gives the
            ``1/(|S|-1)`` of the MATLAB code.
        eig_floor: constant added to the covariance eigenvalues before the square root
            (``1e-7`` in the MATLAB code; not mentioned in the paper).

    ABS (Eq. 14):
        abs_reference: ``"best"`` uses ``X_best`` as printed in Eq. (14); ``"roulette"`` uses
            the roulette-selected ``X_r`` as the MATLAB code does.
        abs_gate: ``"shared"`` reuses the ``r1`` of Eq. (13) as Eq. (14) is printed;
            ``"independent"`` draws a separate per-agent random number as the MATLAB code does.

    SPDM (Section 3.3):
        nvol_threshold: stagnation threshold on ``nVOL``. The paper states ``0.01`` in the
            text of Section 3.3 and ``0.1`` in Algorithm 1 and Fig. 2; the MATLAB code uses
            ``0.001``.
        stagnation_factor: an agent is stagnant when ``Count_i > stagnation_factor * dim``
            (``2`` in the paper and the code).
        spdm_skip_best: do not re-sample the current best agent (MATLAB code; the paper is
            silent).

    Accounting:
        accounting: ``"strict"`` counts every objective call, including the initial population
            and the SPDM candidates, against ``max_evals``. ``"matlab-as-shipped"`` reproduces
            the counter of ``matlab/MRIME_RL_Cov.m`` (initial population not counted, each
            main-loop evaluation advances the counter with probability
            ``shipped_counter_probability``, SPDM evaluations not counted); it performs roughly
            ten times ``max_evals`` objective calls and exists only as a diagnostic.
        shipped_counter_probability: ``0.1`` in ``matlab/MRIME_RL_Cov.m`` line 164.
    """

    use_gcls: bool = True
    use_abs: bool = True
    use_spdm: bool = True

    w: int = 5
    theta_mode: ThetaMode = "paper"

    elite_fraction: float = 0.3
    archive_capacity_per_dim: int = 25
    neighbour_fraction: float = 0.5

    weight_offset: float = 1.0
    weight_order: WeightOrder = "fitness"
    cov_ddof: int = 0
    eig_floor: float = 1e-7

    abs_reference: AbsReference = "best"
    abs_gate: AbsGate = "shared"

    nvol_threshold: float = 0.01
    stagnation_factor: float = 2.0
    spdm_skip_best: bool = True

    accounting: Accounting = "strict"
    shipped_counter_probability: float = 0.1

    def __post_init__(self) -> None:
        if self.w < 1:
            raise ValueError("w must be a positive integer")
        if self.theta_mode not in ("paper", "matlab"):
            raise ValueError(f"unknown theta_mode {self.theta_mode!r}")
        if self.weight_order not in ("fitness", "distance"):
            raise ValueError(f"unknown weight_order {self.weight_order!r}")
        if self.abs_reference not in ("best", "roulette"):
            raise ValueError(f"unknown abs_reference {self.abs_reference!r}")
        if self.abs_gate not in ("shared", "independent"):
            raise ValueError(f"unknown abs_gate {self.abs_gate!r}")
        if self.accounting not in ("strict", "matlab-as-shipped"):
            raise ValueError(f"unknown accounting {self.accounting!r}")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 < self.neighbour_fraction <= 1.0:
            raise ValueError("neighbour_fraction must be in (0, 1]")
        if self.archive_capacity_per_dim < 1:
            raise ValueError("archive_capacity_per_dim must be >= 1")
        if self.weight_offset <= 0.0:
            raise ValueError("weight_offset must be positive")
        if self.cov_ddof not in (0, 1):
            raise ValueError("cov_ddof must be 0 or 1")
        if self.eig_floor < 0.0:
            raise ValueError("eig_floor must be non-negative")
        if not 0.0 < self.shipped_counter_probability <= 1.0:
            raise ValueError("shipped_counter_probability must be in (0, 1]")
        if self.accounting == "matlab-as-shipped" and not (
            self.use_gcls and self.use_abs and self.use_spdm
        ):
            raise ValueError(
                "accounting='matlab-as-shipped' reproduces matlab/MRIME_RL_Cov.m and is only "
                "defined for the full MRIME-CD (all three strategies enabled)"
            )

    @property
    def needs_dominant_group(self) -> bool:
        """Whether ``X_mean`` (and therefore the archive) is needed at all."""
        return self.use_gcls or self.use_abs or self.use_spdm

    @property
    def needs_covariance(self) -> bool:
        """Whether the covariance matrix ``C`` is needed (GCLS or SPDM)."""
        return self.use_gcls or self.use_spdm


# Compatibility profiles -------------------------------------------------------------------

#: Equations exactly as printed in the paper (default).
PAPER = MRIMEConfig()

#: Operator-level behaviour of the author's MATLAB files, with strict evaluation accounting.
MATLAB = replace(
    PAPER,
    theta_mode="matlab",
    weight_offset=0.5,
    weight_order="distance",
    cov_ddof=1,
    abs_reference="roulette",
    abs_gate="independent",
    nvol_threshold=1e-3,
)

#: ``MATLAB`` plus the evaluation counter of ``matlab/MRIME_RL_Cov.m`` (diagnostic only).
MATLAB_AS_SHIPPED = replace(MATLAB, accounting="matlab-as-shipped")

PROFILES: dict[str, MRIMEConfig] = {
    "paper": PAPER,
    "matlab": MATLAB,
    "matlab-as-shipped": MATLAB_AS_SHIPPED,
}

#: Strategy switches of the ablation study, paper Table 5: name -> (GCLS, ABS, SPDM).
VARIANTS: dict[str, tuple[bool, bool, bool]] = {
    "rime": (False, False, False),
    "rime-g": (True, False, False),
    "rime-a": (False, True, False),
    "rime-s": (False, False, True),
    "rime-ga": (True, True, False),
    "rime-gs": (True, False, True),
    "rime-as": (False, True, True),
    "mrime-cd": (True, True, True),
}

#: MATLAB file that implements each variant in the author's archive.
MATLAB_FILES: dict[str, str] = {
    "rime": "RIME.m",
    "rime-g": "MRIME_RL_1.m",
    "rime-a": "MRIME_RL_2.m",
    "rime-s": "MRIME_RL_3.m",
    "rime-ga": "MRIME_RL_12.m",
    "rime-gs": "MRIME_RL_13.m",
    "rime-as": "MRIME_RL_23.m",
    "mrime-cd": "MRIME_RL_Cov.m",
}


def make_config(variant: str = "mrime-cd", profile: str = "paper", **overrides) -> MRIMEConfig:
    """Build the configuration of a named variant under a compatibility profile.

    Args:
        variant: one of :data:`VARIANTS` (case-insensitive), e.g. ``"mrime-cd"`` or ``"rime-ga"``.
        profile: one of :data:`PROFILES`: ``"paper"`` (default), ``"matlab"`` or
            ``"matlab-as-shipped"``.
        **overrides: individual :class:`MRIMEConfig` fields to replace.
    """
    key = variant.lower()
    if key not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    gcls, abs_, spdm = VARIANTS[key]
    fields = {"use_gcls": gcls, "use_abs": abs_, "use_spdm": spdm, **overrides}
    return replace(PROFILES[profile], **fields)


def parse_algorithm(spec: str) -> tuple[str, str]:
    """Split an algorithm spec ``"<variant>[@<profile>]"`` into ``(variant, profile)``.

    Examples: ``"mrime-cd"`` -> ``("mrime-cd", "paper")``; ``"rime@matlab"`` ->
    ``("rime", "matlab")``.
    """
    variant, _, profile = spec.partition("@")
    variant = variant.strip().lower()
    profile = profile.strip().lower() or "paper"
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    return variant, profile
