"""
Quantum Wave Packet Dynamics
============================
Numerical simulation of 1D quantum wave packet evolution using the
Split-Step Fourier Method (SSFM). Supports free propagation, potential
barriers, and harmonic traps. Computes both position and momentum space
representations at each timestep.

"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ─────────────────────────────────────────────
# Physical constants (atomic units: ℏ = m = 1)
# ─────────────────────────────────────────────
HBAR = 1.0
MASS = 1.0


@dataclass
class GridConfig:
    """Spatial and temporal grid parameters."""
    N: int = 1024              # number of spatial grid points (power of 2 for FFT)
    x_min: float = -20.0
    x_max: float = 20.0
    t_max: float = 8.0
    dt: float = 0.005
    store_every: int = 10      # store snapshot every N steps

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.N

    @property
    def x(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.N, endpoint=False)

    @property
    def k(self) -> np.ndarray:
        """Momentum-space grid (via fftfreq)."""
        return 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)

    @property
    def n_steps(self) -> int:
        return int(self.t_max / self.dt)


@dataclass
class WavePacketConfig:
    """Initial Gaussian wave packet parameters."""
    x0: float = -7.0     # initial centre position
    k0: float = 3.0      # initial mean momentum (carrier wave)
    sigma: float = 1.0   # initial spatial width


@dataclass
class SimulationResult:
    """Container for simulation output."""
    times: np.ndarray
    x: np.ndarray
    k: np.ndarray
    psi_x: list          # list of position-space wavefunctions
    psi_k: list          # list of momentum-space wavefunctions
    prob_x: list         # |ψ(x)|²
    prob_k: list         # |ψ(k)|²
    potential: np.ndarray
    grid: GridConfig
    packet: WavePacketConfig
    scenario: str


def gaussian_packet(x: np.ndarray, cfg: WavePacketConfig) -> np.ndarray:
    """Normalised Gaussian wave packet."""
    envelope = np.exp(-((x - cfg.x0) ** 2) / (4 * cfg.sigma ** 2))
    carrier  = np.exp(1j * cfg.k0 * x)
    psi      = envelope * carrier
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * (x[1] - x[0]))


# ─────────────────────────────────────────
# Potential energy profiles
# ─────────────────────────────────────────

def potential_free(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def potential_barrier(x: np.ndarray,
                      center: float = 0.0,
                      width: float = 0.5,
                      height: float = 5.0) -> np.ndarray:
    V = np.zeros_like(x)
    V[np.abs(x - center) < width / 2] = height
    return V


def potential_double_barrier(x: np.ndarray) -> np.ndarray:
    return (potential_barrier(x, center=-1.0, width=0.4, height=6.0) +
            potential_barrier(x,  center= 1.0, width=0.4, height=6.0))


def potential_harmonic(x: np.ndarray, omega: float = 0.5) -> np.ndarray:
    return 0.5 * MASS * omega ** 2 * x ** 2


# ─────────────────────────────────────────
# Split-Step Fourier Method (SSFM)
# ─────────────────────────────────────────

def evolve(grid: GridConfig,
           packet: WavePacketConfig,
           potential_fn: Callable,
           scenario: str = "free") -> SimulationResult:
    """
    Evolve a Gaussian wave packet under potential_fn using SSFM.

    The time-evolution operator is split as:
        U(dt) ≈ exp(-iV·dt/2ℏ) · FFT⁻¹[ exp(-iK·dt/ℏ) · FFT[ exp(-iV·dt/2ℏ) ψ ] ]

    This is 2nd-order accurate (Strang splitting).
    """
    x = grid.x
    k = grid.k
    dx = grid.dx
    dt = grid.dt

    V = potential_fn(x).astype(complex)
    K = HBAR * k ** 2 / (2 * MASS)   # kinetic energy in k-space

    # Pre-compute propagators
    half_V_prop = np.exp(-1j * V * dt / (2 * HBAR))
    full_K_prop = np.exp(-1j * K * dt / HBAR)

    # Initialise wave function
    psi = gaussian_packet(x, packet).astype(complex)

    # Storage
    times, psi_x_list, psi_k_list, prob_x_list, prob_k_list = [], [], [], [], []

    def snapshot(t, psi):
        psi_k = np.fft.fftshift(np.fft.fft(psi)) * dx / np.sqrt(2 * np.pi)
        k_shifted = np.fft.fftshift(k)
        times.append(t)
        psi_x_list.append(psi.copy())
        psi_k_list.append(psi_k.copy())
        prob_x_list.append(np.abs(psi) ** 2)
        prob_k_list.append(np.abs(psi_k) ** 2)

    snapshot(0.0, psi)

    for step in range(1, grid.n_steps + 1):
        # Half step in position space
        psi = half_V_prop * psi
        # Full step in momentum space
        psi_k = np.fft.fft(psi)
        psi_k = full_K_prop * psi_k
        psi = np.fft.ifft(psi_k)
        # Half step in position space
        psi = half_V_prop * psi

        if step % grid.store_every == 0:
            snapshot(step * dt, psi)

    k_shifted = np.fft.fftshift(k)

    return SimulationResult(
        times      = np.array(times),
        x          = x,
        k          = k_shifted,
        psi_x      = psi_x_list,
        psi_k      = psi_k_list,
        prob_x     = prob_x_list,
        prob_k     = prob_k_list,
        potential  = V.real,
        grid       = grid,
        packet     = packet,
        scenario   = scenario,
    )


# ─────────────────────────────────────────
# Pre-built scenarios
# ─────────────────────────────────────────

def run_free_particle() -> SimulationResult:
    grid   = GridConfig(t_max=8.0, dt=0.005, store_every=10)
    packet = WavePacketConfig(x0=-7.0, k0=3.0, sigma=1.0)
    return evolve(grid, packet, potential_free, scenario="Free Particle")


def run_tunneling() -> SimulationResult:
    grid   = GridConfig(t_max=6.0, dt=0.005, store_every=8)
    packet = WavePacketConfig(x0=-6.0, k0=3.0, sigma=0.8)
    return evolve(grid, packet, potential_barrier, scenario="Quantum Tunneling")


def run_double_barrier() -> SimulationResult:
    """Resonant tunneling — transmission peaks at resonant energies."""
    grid   = GridConfig(t_max=8.0, dt=0.005, store_every=10)
    packet = WavePacketConfig(x0=-7.0, k0=2.5, sigma=1.2)
    return evolve(grid, packet, potential_double_barrier, scenario="Resonant Tunneling")


def run_harmonic() -> SimulationResult:
    grid   = GridConfig(t_max=15.0, dt=0.005, store_every=15)
    packet = WavePacketConfig(x0=3.0, k0=0.0, sigma=1.0)
    return evolve(grid, packet, potential_harmonic, scenario="Harmonic Oscillator")


if __name__ == "__main__":
    print("Running all scenarios…")
    for fn, name in [(run_free_particle,  "free"),
                     (run_tunneling,      "tunnel"),
                     (run_double_barrier, "double"),
                     (run_harmonic,       "harmonic")]:
        r = fn()
        norm = np.trapezoid(r.prob_x[-1], r.x)
        print(f"  {r.scenario:30s}  steps={len(r.times)}  norm={norm:.6f}")
    print("All scenarios complete.")
