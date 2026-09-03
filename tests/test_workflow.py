"""Representative QuSpin workflow using an installed sidecar rule."""

from __future__ import annotations

import numpy as np

import quspin_ad  # noqa: F401 - explicit registration
import chainrules as ad
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import ED_state_vs_time


def test_spin_chain_eigensystem_to_ad_time_evolution() -> None:
    """Build a small spin chain, diagonalize it, then differentiate evolution."""
    basis = spin_basis_1d(3)
    couplings = [[1.0, i, (i + 1) % 3] for i in range(3)]
    hamiltonian_object = hamiltonian(
        [["zz", couplings]], [], basis=basis, dtype=np.float64
    )
    energy, eigenvectors = hamiltonian_object.eigh()
    initial = np.zeros(hamiltonian_object.Ns)
    initial[0] = 1.0
    times = np.linspace(0.0, 0.5, 4)
    value, tangent = ad.jvp(
        ED_state_vs_time,
        initial,
        energy,
        eigenvectors,
        times,
        tangents={"times": np.ones_like(times)},
    )
    assert value.shape == (hamiltonian_object.Ns, times.size)
    assert tangent.shape == value.shape
    assert np.all(np.isfinite(value))
    assert np.all(np.isfinite(tangent))
