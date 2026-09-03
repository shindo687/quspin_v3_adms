from __future__ import annotations

import numpy as np
import pytest

import quspin_ad  # noqa: F401 - explicit rule registration
import chainrules as ad
from quspin.basis import coherent_state as native_coherent_state
from quspin.operators import anti_commutator, commutator
from quspin.tools.evolution import ED_state_vs_time
from quspin.tools.lanczos import lin_comb_Q_T
from quspin.tools.misc import KL_div
from quspin.tools.misc import project_op


def central(fun, x, dx, eps=1e-6):
    return (fun(x + eps * dx) - fun(x - eps * dx)) / (2 * eps)


def test_kl_primal_jvp_vjp_and_duality() -> None:
    p1 = np.array([0.2, 0.3, 0.5])
    p2 = np.array([0.3, 0.3, 0.4])
    dp1 = np.array([0.1, -0.04, -0.06])
    dp2 = np.array([-0.02, 0.01, 0.01])
    value, tangent = ad.jvp(KL_div, p1, p2, tangents={"p1": dp1, "p2": dp2})
    oracle = central(lambda x: KL_div(x, p2), p1, dp1) + central(
        lambda x: KL_div(p1, x), p2, dp2
    )
    assert value == KL_div(p1, p2)
    assert np.allclose(tangent, oracle, rtol=2e-6, atol=2e-6)
    value, pullback = ad.vjp(KL_div, p1, p2, wrt=("p1", "p2"))
    cotangent = 1.7
    grads = pullback(cotangent)
    assert np.allclose(grads["p1"], cotangent * (np.log(p1 / p2) + 1))
    assert np.allclose(grads["p2"], -cotangent * p1 / p2)
    assert np.allclose(
        np.real(np.vdot(grads["p1"], dp1) + np.vdot(grads["p2"], dp2)),
        np.real(np.conj(cotangent) * tangent),
    )
    assert pullback(ad.ZERO) == {"p1": ad.ZERO, "p2": ad.ZERO}
    zero_value, zero_tangent = ad.jvp(KL_div, p1, p2, tangents={"p1": ad.ZERO})
    assert zero_value == value
    assert zero_tangent is ad.ZERO


def test_kl_grad_and_value_and_grad_are_vjp_adapters() -> None:
    p1 = np.array([0.25, 0.35, 0.40])
    p2 = np.array([0.30, 0.30, 0.40])
    gradients = ad.grad(KL_div, p1, p2, wrt=("p1", "p2"))
    value, same_gradients = ad.value_and_grad(KL_div, p1, p2, wrt=("p1", "p2"))
    assert value == KL_div(p1, p2)
    assert np.array_equal(gradients["p1"], same_gradients["p1"])
    assert np.array_equal(gradients["p2"], same_gradients["p2"])


def test_kl_invalid_domain_is_preserved() -> None:
    with pytest.raises((TypeError, ValueError)):
        ad.jvp(
            KL_div,
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
            tangents={"p1": np.ones(2)},
        )


def test_coherent_state_real_and_complex_oracle() -> None:
    a = 0.8 + 0.4j
    da = 0.2 - 0.7j
    value, tangent = ad.jvp(
        native_coherent_state, a, 6, dtype=np.complex128, tangents={"a": da}
    )
    oracle = central(lambda x: native_coherent_state(x, 6, dtype=np.complex128), a, da)
    assert np.allclose(value, native_coherent_state(a, 6, dtype=np.complex128))
    assert np.allclose(tangent, oracle, rtol=5e-6, atol=5e-6)
    cotangent = np.arange(6) + 1j * np.arange(1, 7)
    _, pullback = ad.vjp(native_coherent_state, a, 6, dtype=np.complex128, wrt="a")
    grad = pullback(cotangent)["a"]
    assert np.allclose(
        np.real(np.conj(grad) * da), np.real(np.vdot(cotangent, tangent))
    )
    with pytest.raises(ad.NonDifferentiablePoint, match="a=0"):
        ad.jvp(native_coherent_state, 0.0, 4, tangents={"a": 1.0})


def test_matrix_rules_oracle_and_duality() -> None:
    rng = np.random.default_rng(4)
    h1 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    h2 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    d1 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    d2 = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    cotangent = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    for function in (commutator, anti_commutator):
        value, tangent = ad.jvp(function, h1, h2, tangents={"H1": d1, "H2": d2})
        oracle = central(lambda x: function(x, h2), h1, d1) + central(
            lambda x: function(h1, x), h2, d2
        )
        assert np.allclose(value, function(h1, h2))
        assert np.allclose(tangent, oracle, rtol=2e-6, atol=2e-6)
        _, pullback = ad.vjp(function, h1, h2, wrt=("H1", "H2"))
        grads = pullback(cotangent)
        lhs = np.real(np.vdot(cotangent, tangent))
        rhs = np.real(np.vdot(grads["H1"], d1) + np.vdot(grads["H2"], d2))
        assert np.allclose(lhs, rhs, rtol=2e-6, atol=2e-6)
        assert pullback(ad.ZERO) == {"H1": ad.ZERO, "H2": ad.ZERO}


def test_ed_state_vs_time_oracle_and_active_inputs() -> None:
    rng = np.random.default_rng(9)
    n, nt = 3, 4
    q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    energy = np.array([0.2, 0.7, 1.1])
    times = np.linspace(0.1, 0.7, nt)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    dpsi = rng.normal(size=n) + 1j * rng.normal(size=n)
    denergy = rng.normal(size=n)
    dtimes = rng.normal(size=nt)
    value, tangent = ad.jvp(
        ED_state_vs_time,
        psi,
        energy,
        q,
        times,
        tangents={"psi": dpsi, "E": denergy, "times": dtimes},
    )
    # Use a complete central perturbation of all three active inputs.
    eps = 1e-6
    oracle = (
        ED_state_vs_time(
            psi + eps * dpsi, energy + eps * denergy, q, times + eps * dtimes
        )
        - ED_state_vs_time(
            psi - eps * dpsi, energy - eps * denergy, q, times - eps * dtimes
        )
    ) / (2 * eps)
    assert value.shape == (n, nt)
    assert np.allclose(tangent, oracle, rtol=2e-5, atol=2e-5)
    cotangent = rng.normal(size=(n, nt)) + 1j * rng.normal(size=(n, nt))
    _, pullback = ad.vjp(
        ED_state_vs_time, psi, energy, q, times, wrt=("psi", "E", "times")
    )
    grads = pullback(cotangent)
    assert np.allclose(
        np.real(np.vdot(cotangent, tangent)),
        np.real(
            np.vdot(grads["psi"], dpsi)
            + np.vdot(grads["E"], denergy)
            + np.vdot(grads["times"], dtimes)
        ),
        rtol=2e-5,
        atol=2e-5,
    )
    zero_value, zero_tangent = ad.jvp(
        ED_state_vs_time, psi, energy, q, times, tangents={"E": ad.ZERO}
    )
    assert np.array_equal(zero_value, value)
    assert zero_tangent is ad.ZERO
    with pytest.raises(ad.NonDifferentiablePoint, match="iterate=False"):
        ad.vjp(ED_state_vs_time, psi, energy, q, times, iterate=True, wrt="psi")


def test_lin_comb_qt_oracle_and_vjp() -> None:
    rng = np.random.default_rng(12)
    coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    q_t = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    dcoeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    dq = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    value, tangent = ad.jvp(
        lin_comb_Q_T, coeff, q_t, tangents={"coeff": dcoeff, "Q_T": dq}
    )
    eps = 1e-6
    oracle = (
        lin_comb_Q_T(coeff + eps * dcoeff, q_t + eps * dq)
        - lin_comb_Q_T(coeff - eps * dcoeff, q_t - eps * dq)
    ) / (2 * eps)
    assert np.allclose(value, coeff @ q_t)
    assert np.allclose(tangent, oracle, rtol=2e-6, atol=2e-6)
    cotangent = rng.normal(size=5) + 1j * rng.normal(size=5)
    _, pullback = ad.vjp(lin_comb_Q_T, coeff, q_t, wrt=("coeff", "Q_T"))
    grads = pullback(cotangent)
    assert np.allclose(
        np.real(np.vdot(cotangent, tangent)),
        np.real(np.vdot(grads["coeff"], dcoeff) + np.vdot(grads["Q_T"], dq)),
    )
    assert pullback(ad.ZERO) == {"coeff": ad.ZERO, "Q_T": ad.ZERO}


def test_project_op_dense_structured_rule() -> None:
    """Check dense projection in both primal and reverse modes."""
    rng = np.random.default_rng(22)
    observable = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    projector = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    d_observable = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    d_projector = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    value, tangent = ad.jvp(
        project_op,
        observable,
        projector,
        tangents={"Obs": d_observable, "proj": d_projector},
    )
    eps = 1e-6
    plus = project_op(observable + eps * d_observable, projector + eps * d_projector)
    minus = project_op(observable - eps * d_observable, projector - eps * d_projector)
    oracle = {"Proj_Obs": (plus["Proj_Obs"] - minus["Proj_Obs"]) / (2 * eps)}
    assert np.allclose(value["Proj_Obs"], project_op(observable, projector)["Proj_Obs"])
    assert np.allclose(tangent["Proj_Obs"], oracle["Proj_Obs"], rtol=2e-6, atol=2e-6)
    cotangent = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    _, pullback = ad.vjp(project_op, observable, projector, wrt=("Obs", "proj"))
    gradients = pullback({"Proj_Obs": cotangent})
    lhs = np.real(np.vdot(cotangent, tangent["Proj_Obs"]))
    rhs = np.real(
        np.vdot(gradients["Obs"], d_observable)
        + np.vdot(gradients["proj"], d_projector)
    )
    assert np.allclose(lhs, rhs, rtol=2e-6, atol=2e-6)
    assert pullback(ad.ZERO) == {"Obs": ad.ZERO, "proj": ad.ZERO}
