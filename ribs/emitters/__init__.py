"""Emitters output new candidate solutions in QD algorithms.

.. note::
    Emitters provided here take on the data type of the archive passed to their
    constructor. For instance, if an archive has dtype ``np.float64``, then an
    emitter created with that archive will emit solutions with dtype
    ``np.float64``.

.. autosummary::
    :toctree:

    ribs.emitters.AnnealingEmitter
    ribs.emitters.GaussianEmitter
    ribs.emitters.GradientEmitter
    ribs.emitters.GradientImprovementEmitter
    ribs.emitters.IsoLineEmitter
    ribs.emitters.ImprovementEmitter
    ribs.emitters.RandomDirectionEmitter
    ribs.emitters.OptimizingEmitter
    ribs.emitters.EmitterBase
"""
from ribs.emitters._annealing_emitter import AnnealingEmitter
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters._gaussian_emitter import GaussianEmitter
from ribs.emitters._gradient_emitter import GradientEmitter
from ribs.emitters._gradient_annealing_emitter import GradientAnnealingEmitter
from ribs.emitters._gradient_improvement_emitter import GradientImprovementEmitter
from ribs.emitters._improvement_emitter import ImprovementEmitter
from ribs.emitters._iso_line_emitter import IsoLineEmitter
from ribs.emitters._optimizing_emitter import OptimizingEmitter
from ribs.emitters._random_direction_emitter import RandomDirectionEmitter

__all__ = [
    "AnnealingEmitter",
    "GaussianEmitter",
    "GradientEmitter",
    "GradientImprovementEmitter",
    "IsoLineEmitter",
    "ImprovementEmitter",
    "RandomDirectionEmitter",
    "OptimizingEmitter",
    "EmitterBase",
]
