"""
Visualization module for quantum wave packet simulations.
Produces publication-quality plots and animations.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from simulation import SimulationResult, run_free_particle, run_tunneling, run_double_barrier, run_harmonic

# ─────────────────────────────────────────
# Aesthetic configuration
# ─────────────────────────────────────────

DARK_BG    = "#050810"
PANEL_BG   = "#0a0f1e"
GRID_COLOR = "#1a2340"
TEXT_COLOR = "#c8d8f0"
DIM_TEXT   = "#4a6080"

WAVE_CYAN   = "#00e5ff"
WAVE_PURPLE = "#b44fff"
WAVE_ORANGE = "#ff8c42"
POTENTIAL_Y = "#f5c542"
ACCENT_PINK = "#ff3fa4"

FONT_TITLE = "DejaVu Sans"

# Custom colormaps
_wave_cmap = LinearSegmentedColormap.from_list(
    "wave", ["#050810", "#0a1f4e", "#0066cc", "#00e5ff", "#ffffff"])
_prob_cmap = LinearSegmentedColormap.from_list(
    "prob", ["#050810", "#1a0a2e", "#6600cc", "#b44fff", "#ffffff"])


def _apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=DIM_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, alpha=0.6)
    if title:  ax.set_title(title, color=TEXT_COLOR, fontsize=9, pad=6, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, color=DIM_TEXT, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=DIM_TEXT, fontsize=8)


# ─────────────────────────────────────────
# 1. Static snapshot overview
# ─────────────────────────────────────────

def plot_snapshot_overview(results: list[SimulationResult],
                            save_path: str = "plots/01_snapshot_overview.png"):
    Path(save_path).parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    fig.suptitle("Quantum Wave Packet Dynamics — Snapshot Overview",
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', y=0.97)

    n_scenarios = len(results)
    n_snaps = 3  # early / mid / late

    outer = gridspec.GridSpec(n_scenarios, 1, figure=fig, hspace=0.55)

    for row, res in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(1, n_snaps, subplot_spec=outer[row], wspace=0.3)
        indices = [0, len(res.times)//2, -1]
        labels  = ["Initial", "Mid evolution", "Final state"]

        for col, (idx, lbl) in enumerate(zip(indices, labels)):
            ax = fig.add_subplot(inner[col])
            x  = res.x

            # Potential (scaled for visibility)
            V_norm = res.potential / (res.potential.max() + 1e-9)
            ax.fill_between(x, V_norm * 0.4, alpha=0.25, color=POTENTIAL_Y)
            ax.plot(x, V_norm * 0.4, color=POTENTIAL_Y, linewidth=0.8, alpha=0.5)

            # Probability density
            prob = res.prob_x[idx]
            prob_norm = prob / (prob.max() + 1e-12)
            ax.fill_between(x, prob_norm, alpha=0.35, color=WAVE_CYAN)
            ax.plot(x, prob_norm, color=WAVE_CYAN, linewidth=1.2)

            # Real part
            psi_re = res.psi_x[idx].real
            psi_re_norm = psi_re / (np.abs(psi_re).max() + 1e-12) * 0.8
            ax.plot(x, psi_re_norm, color=WAVE_PURPLE, linewidth=0.7, alpha=0.7)

            t_val = res.times[idx]
            _apply_style(ax, title=f"{lbl}  (t={t_val:.1f})", xlabel="x (a.u.)")
            ax.set_ylim(-1, 1.2)
            ax.set_xlim(x[0], x[-1])

            if col == 0:
                ax.set_ylabel(res.scenario, color=WAVE_ORANGE, fontsize=8, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=WAVE_CYAN,   lw=2, label="|ψ(x)|²"),
        Line2D([0],[0], color=WAVE_PURPLE, lw=1.5, alpha=0.8, label="Re[ψ(x)]"),
        Line2D([0],[0], color=POTENTIAL_Y, lw=1.5, alpha=0.7, label="V(x) scaled"),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# 2. Phase-space density heatmap (x–t)
# ─────────────────────────────────────────

def plot_spacetime_heatmap(res: SimulationResult,
                            save_path: str = "plots/02_spacetime.png"):
    Path(save_path).parent.mkdir(exist_ok=True)

    prob_matrix = np.array(res.prob_x).T   # shape: (N_x, N_t)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle(f"Space–Time Probability Density  |  {res.scenario}",
                 color=TEXT_COLOR, fontsize=12, fontweight='bold')

    # Position space
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    extent = [res.times[0], res.times[-1], res.x[0], res.x[-1]]
    im = ax.imshow(prob_matrix, aspect='auto', origin='lower',
                   extent=extent, cmap=_prob_cmap, interpolation='bilinear')
    # Overlay potential as contour
    if res.potential.max() > 0:
        V_line = res.potential / res.potential.max()
        for t in res.times[::max(1, len(res.times)//20)]:
            ax.plot([t]*2, [res.x[V_line > 0.5][0] if (V_line > 0.5).any() else res.x[0],
                            res.x[V_line > 0.5][-1] if (V_line > 0.5).any() else res.x[-1]],
                    color=POTENTIAL_Y, linewidth=0.4, alpha=0.3)
    cb = plt.colorbar(im, ax=ax, fraction=0.03)
    cb.ax.tick_params(colors=DIM_TEXT, labelsize=7)
    cb.set_label("|ψ(x,t)|²", color=DIM_TEXT, fontsize=8)
    _apply_style(ax, title="Position Space  |ψ(x,t)|²", xlabel="time t (a.u.)", ylabel="position x (a.u.)")

    # Momentum space
    ax = axes[1]
    ax.set_facecolor(DARK_BG)
    prob_k_matrix = np.array(res.prob_k).T
    # Zoom to relevant k range
    k = res.k
    k_mask = np.abs(k) < 10
    extent_k = [res.times[0], res.times[-1], k[k_mask][0], k[k_mask][-1]]
    im2 = ax.imshow(prob_k_matrix[k_mask], aspect='auto', origin='lower',
                    extent=extent_k, cmap=_wave_cmap, interpolation='bilinear')
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.03)
    cb2.ax.tick_params(colors=DIM_TEXT, labelsize=7)
    cb2.set_label("|ψ̃(k,t)|²", color=DIM_TEXT, fontsize=8)
    _apply_style(ax, title="Momentum Space  |ψ̃(k,t)|²", xlabel="time t (a.u.)", ylabel="momentum k (a.u.)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# 3. Dual-space evolution panel
# ─────────────────────────────────────────

def plot_dual_space(res: SimulationResult,
                    save_path: str = "plots/03_dual_space.png"):
    Path(save_path).parent.mkdir(exist_ok=True)

    n_frames = min(6, len(res.times))
    frame_idx = np.linspace(0, len(res.times)-1, n_frames, dtype=int)

    fig = plt.figure(figsize=(18, 8), facecolor=DARK_BG)
    fig.suptitle(f"Position ↔ Momentum Duality  |  {res.scenario}",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, n_frames, figure=fig, hspace=0.45, wspace=0.25)

    k = res.k
    k_mask = np.abs(k) < 10

    colors_t = plt.cm.plasma(np.linspace(0.2, 0.9, n_frames))

    for col, idx in enumerate(frame_idx):
        t = res.times[idx]
        color = colors_t[col]

        # — Position space —
        ax_x = fig.add_subplot(gs[0, col])
        ax_x.set_facecolor(PANEL_BG)

        prob = res.prob_x[idx]
        V_sc = res.potential / (res.potential.max()+1e-9) * prob.max()
        ax_x.fill_between(res.x, prob, alpha=0.4, color=color)
        ax_x.plot(res.x, prob, color=color, lw=1.3)
        ax_x.fill_between(res.x, V_sc, alpha=0.2, color=POTENTIAL_Y)
        ax_x.plot(res.x, V_sc, color=POTENTIAL_Y, lw=0.7, alpha=0.6)

        ax_x.set_xlim(res.x[0], res.x[-1])
        ax_x.set_ylim(0, None)
        _apply_style(ax_x, title=f"t = {t:.2f}", xlabel="x")
        if col == 0:
            ax_x.set_ylabel("|ψ(x)|²", color=TEXT_COLOR, fontsize=8)
        ax_x.tick_params(labelsize=7)

        # — Momentum space —
        ax_k = fig.add_subplot(gs[1, col])
        ax_k.set_facecolor(PANEL_BG)

        prob_k = res.prob_k[idx][k_mask]
        ax_k.fill_between(k[k_mask], prob_k, alpha=0.4, color=WAVE_PURPLE)
        ax_k.plot(k[k_mask], prob_k, color=WAVE_PURPLE, lw=1.3)

        ax_k.set_xlim(k[k_mask][0], k[k_mask][-1])
        ax_k.set_ylim(0, None)
        _apply_style(ax_k, xlabel="k")
        if col == 0:
            ax_k.set_ylabel("|ψ̃(k)|²", color=TEXT_COLOR, fontsize=8)
        ax_k.tick_params(labelsize=7)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# 4. Tunneling analysis — transmission coefficient
# ─────────────────────────────────────────

def plot_tunneling_analysis(res: SimulationResult,
                             save_path: str = "plots/04_tunneling_analysis.png"):
    Path(save_path).parent.mkdir(exist_ok=True)

    # Compute transmitted vs reflected probability over time
    barrier_x = 0.0
    x = res.x
    left_mask  = x < barrier_x
    right_mask = x > barrier_x

    dx = x[1] - x[0]
    trans = [np.sum(p[right_mask]) * dx for p in res.prob_x]
    refl  = [np.sum(p[left_mask])  * dx for p in res.prob_x]
    total = [t + r for t, r in zip(trans, refl)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(f"Tunneling Analysis  |  {res.scenario}",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    # — Probability flow —
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.plot(res.times, trans, color=WAVE_CYAN,   lw=1.8, label="Transmitted")
    ax.plot(res.times, refl,  color=ACCENT_PINK, lw=1.8, label="Reflected")
    ax.plot(res.times, total, color=TEXT_COLOR,  lw=1.0, ls='--', alpha=0.5, label="Total (norm)")
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    _apply_style(ax, title="Probability Flow", xlabel="time (a.u.)", ylabel="probability")

    # — Final spatial distribution —
    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    prob_final = res.prob_x[-1]
    V_sc = res.potential / (res.potential.max()+1e-9) * prob_final.max() * 0.8
    ax.fill_between(x[left_mask],  prob_final[left_mask],  alpha=0.5, color=ACCENT_PINK)
    ax.fill_between(x[right_mask], prob_final[right_mask], alpha=0.5, color=WAVE_CYAN)
    ax.plot(x, prob_final, color=TEXT_COLOR, lw=0.8, alpha=0.6)
    ax.fill_between(x, V_sc, alpha=0.3, color=POTENTIAL_Y)
    ax.plot(x, V_sc, color=POTENTIAL_Y, lw=1.0)
    ax.axvline(barrier_x, color=POTENTIAL_Y, lw=0.8, ls='--', alpha=0.6)
    _apply_style(ax, title="Final |ψ(x)|²", xlabel="x (a.u.)", ylabel="|ψ|²")

    # — Momentum distribution comparison —
    ax = axes[2]
    ax.set_facecolor(PANEL_BG)
    k = res.k
    k_mask = np.abs(k) < 12
    ax.fill_between(k[k_mask], res.prob_k[0][k_mask],  alpha=0.4, color=DIM_TEXT,   label="Initial")
    ax.fill_between(k[k_mask], res.prob_k[-1][k_mask], alpha=0.5, color=WAVE_PURPLE, label="Final")
    ax.plot(k[k_mask], res.prob_k[0][k_mask],  color=DIM_TEXT,    lw=1.2)
    ax.plot(k[k_mask], res.prob_k[-1][k_mask], color=WAVE_PURPLE, lw=1.2)
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    _apply_style(ax, title="Momentum Distribution", xlabel="k (a.u.)", ylabel="|ψ̃(k)|²")

    T_final = trans[-1]
    R_final = refl[-1]
    fig.text(0.5, 0.01,
             f"Transmission T = {T_final:.3f}   |   Reflection R = {R_final:.3f}   |   T+R = {T_final+R_final:.4f}",
             ha='center', color=WAVE_ORANGE, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0,0.04,1,1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# 5. Uncertainty principle verification
# ─────────────────────────────────────────

def plot_uncertainty(results: list[SimulationResult],
                      save_path: str = "plots/05_uncertainty.png"):
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("Heisenberg Uncertainty Principle  Δx · Δk ≥ ½",
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    colors = [WAVE_CYAN, WAVE_PURPLE, WAVE_ORANGE, ACCENT_PINK]

    ax_prod = axes[0]
    ax_sigx = axes[1]
    ax_prod.set_facecolor(PANEL_BG)
    ax_sigx.set_facecolor(PANEL_BG)

    for res, c in zip(results, colors):
        dx = res.x[1] - res.x[0]
        sig_x, sig_k, prod = [], [], []

        for prob_x, prob_k in zip(res.prob_x, res.prob_k):
            # <x> and Δx
            mean_x  = np.sum(res.x * prob_x) * dx
            mean_x2 = np.sum(res.x**2 * prob_x) * dx
            sx = np.sqrt(max(mean_x2 - mean_x**2, 0))

            # <k> and Δk  (using k-space probability)
            dk = res.k[1] - res.k[0]
            norm_k = np.sum(prob_k) * dk
            prob_k_n = prob_k / (norm_k + 1e-12)
            mean_k  = np.sum(res.k * prob_k_n) * dk
            mean_k2 = np.sum(res.k**2 * prob_k_n) * dk
            sk = np.sqrt(max(mean_k2 - mean_k**2, 0))

            sig_x.append(sx)
            sig_k.append(sk)
            prod.append(sx * sk)

        ax_prod.plot(res.times, prod, color=c, lw=1.5, label=res.scenario)
        ax_sigx.plot(res.times, sig_x, color=c, lw=1.5, label=f"Δx – {res.scenario}")

    ax_prod.axhline(0.5, color=POTENTIAL_Y, lw=1.2, ls='--', alpha=0.8, label="ℏ/2 = 0.5")
    ax_prod.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)
    _apply_style(ax_prod, title="Uncertainty Product Δx·Δk", xlabel="time (a.u.)", ylabel="Δx · Δk")

    ax_sigx.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7)
    _apply_style(ax_sigx, title="Position Spread Δx(t)", xlabel="time (a.u.)", ylabel="Δx (a.u.)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# 6. GIF animation (tunneling scenario)
# ─────────────────────────────────────────

def make_animation(res: SimulationResult,
                    save_path: str = "plots/06_tunneling_animation.gif",
                    fps: int = 20):
    Path(save_path).parent.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor=DARK_BG)
    fig.suptitle(f"{res.scenario}  —  Wave Packet Evolution",
                 color=TEXT_COLOR, fontsize=12, fontweight='bold')

    ax_x, ax_k = axes
    for ax in axes:
        ax.set_facecolor(PANEL_BG)

    k = res.k
    k_mask = np.abs(k) < 12

    # Static potential
    V_sc = res.potential / (res.potential.max()+1e-9)
    max_prob = max(p.max() for p in res.prob_x)
    V_plot = V_sc * max_prob * 0.6

    ax_x.fill_between(res.x, V_plot, alpha=0.2, color=POTENTIAL_Y)
    ax_x.plot(res.x, V_plot, color=POTENTIAL_Y, lw=1.0, alpha=0.5)

    line_prob, = ax_x.plot([], [], color=WAVE_CYAN,   lw=1.8, label="|ψ(x)|²")
    fill_prob  = ax_x.fill_between([], [], alpha=0)
    line_real, = ax_x.plot([], [], color=WAVE_PURPLE, lw=0.8, alpha=0.7, label="Re[ψ]")
    time_text  = ax_x.text(0.02, 0.92, '', transform=ax_x.transAxes,
                            color=WAVE_ORANGE, fontsize=9, fontweight='bold')

    line_pk, = ax_k.plot([], [], color=WAVE_PURPLE, lw=1.8)
    fill_pk  = ax_k.fill_between([], [], alpha=0)

    ax_x.set_xlim(res.x[0], res.x[-1])
    ax_x.set_ylim(-max_prob*0.9, max_prob*1.15)
    _apply_style(ax_x, xlabel="position x (a.u.)", ylabel="probability density")
    ax_x.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8, loc='upper right')

    max_pk = max(p[k_mask].max() for p in res.prob_k)
    ax_k.set_xlim(k[k_mask][0], k[k_mask][-1])
    ax_k.set_ylim(0, max_pk * 1.15)
    _apply_style(ax_k, xlabel="momentum k (a.u.)", ylabel="|ψ̃(k)|²")

    # Thin out frames for gif size
    frame_indices = list(range(0, len(res.times), max(1, len(res.times)//80)))

    def init():
        line_prob.set_data([], [])
        line_real.set_data([], [])
        line_pk.set_data([], [])
        return line_prob, line_real, line_pk

    def animate(frame_no):
        nonlocal fill_prob, fill_pk
        idx = frame_indices[frame_no]

        prob = res.prob_x[idx]
        psi_re = res.psi_x[idx].real
        psi_re_sc = psi_re / (np.abs(psi_re).max()+1e-12) * max_prob * 0.6

        line_prob.set_data(res.x, prob)
        line_real.set_data(res.x, psi_re_sc)

        for coll in ax_x.collections[1:]:
            coll.remove()
        ax_x.fill_between(res.x, prob, alpha=0.3, color=WAVE_CYAN)

        prob_k = res.prob_k[idx][k_mask]
        line_pk.set_data(k[k_mask], prob_k)
        for coll in ax_k.collections:
            coll.remove()
        ax_k.fill_between(k[k_mask], prob_k, alpha=0.35, color=WAVE_PURPLE)

        time_text.set_text(f"t = {res.times[idx]:.2f} a.u.")
        return line_prob, line_real, line_pk, time_text

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frame_indices), interval=1000//fps, blit=False)
    ani.save(save_path, writer='pillow', fps=fps, dpi=100)
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': False,
    })

    print("Running simulations…")
    free     = run_free_particle()
    tunnel   = run_tunneling()
    double   = run_double_barrier()
    harmonic = run_harmonic()

    all_results = [free, tunnel, double, harmonic]

    print("\nGenerating plots…")
    plot_snapshot_overview(all_results)
    plot_spacetime_heatmap(tunnel)
    plot_dual_space(tunnel)
    plot_tunneling_analysis(tunnel)
    plot_uncertainty(all_results)
    print("\nGenerating animation (this may take ~30s)…")
    make_animation(tunnel)

    print("\n✓ All visualizations saved to plots/")
