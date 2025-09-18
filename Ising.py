import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    return np, plt


@app.cell
def _(mo):
    mo.md(
        """
    # Ising Model Simulation

    This notebook simulates the **2D Ising model** using the **Metropolis algorithm**.

    The Ising model is a fundamental model in statistical physics,
    describing spins $s_i = \\pm 1$ arranged on a lattice.
    Spins interact with their nearest neighbors, and the system
    evolves under thermal fluctuations.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Hamiltonian

    The energy of a spin configuration is given by the Hamiltonian:

    $$
    H = -J \sum_{\langle i, j \rangle} s_i s_j - h \sum_i s_i
    $$

    - $s_i = \pm 1$ is the spin at site $i$.  
    - $\langle i, j \rangle$ denotes nearest-neighbor pairs on the lattice.  
    - $J > 0$ favors alignment of neighboring spins (ferromagnetic case).  
    - $h$ is an external magnetic field (we set $h=0$ in this simulation).  

    In this notebook, we simulate the **ferromagnetic 2D square lattice Ising model**
    with periodic boundary conditions and $J = 1, h = 0$.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The Metropolis Algorithm

    To simulate the thermal fluctuations at temperature $T$, we use the
    **Metropolis–Hastings algorithm**:

    1. Pick a random spin $s_i$.  
    2. Compute the energy change $\Delta E$ if $s_i$ were flipped.  
    3. If $\Delta E \leq 0$, accept the flip.  
    4. Otherwise, accept the flip with probability
       $$
       P = e^{-\Delta E / (k_B T)}
       $$
       (we set $k_B = 1$).  

    Repeating this process constitutes one **Monte Carlo sweep**.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Observables

    - **Magnetisation**  
      $m = \\frac{1}{N} \\sum_i s_i$
      where $N$ is the total number of spins.  

    - **Spin configuration**  
      We visualize the spins on the lattice as black/white pixels.  

    By varying temperature $T$, we can observe a **phase transition**
    near the critical temperature $T_c \\approx 2.27$ for the 2D square lattice Ising model.
    """
    )
    return


@app.cell
def _(np):
    # --- Optimized Simulation core ---
    def run_ising(L, T, n_sweeps, frames_every, seed=42):
        rng = np.random.default_rng(seed)
        spins = rng.choice([1, -1], size=(L, L))

        # Precompute probabilities for ΔE
        boltzmann = {dE: np.exp(-dE / T) for dE in [-8, -4, 0, 4, 8]}

        frames, mag = [], []

        for sweep in range(n_sweeps):
            for parity in [0, 1]:  # checkerboard update
                # Build mask for sublattice
                mask = (np.indices((L, L)).sum(axis=0) % 2 == parity)

                # Compute neighbor sum
                nb = (
                    np.roll(spins, 1, axis=0) +
                    np.roll(spins, -1, axis=0) +
                    np.roll(spins, 1, axis=1) +
                    np.roll(spins, -1, axis=1)
                )

                dE = 2 * spins * nb

                # Decide flips
                rand = rng.random((L, L))
                flip = (rand < np.vectorize(boltzmann.get)(dE)) & mask
                spins[flip] *= -1

            # Track observables
            mag.append(spins.mean())
            if sweep % frames_every == 0:
                frames.append(spins.copy())

        return {"frames": frames, "mag": mag}
    return (run_ising,)


@app.cell
def _(mo):
    controls = mo.ui.dictionary({
        "L": mo.ui.slider(16, 128, value=64, label="System size L", debounce=True),
        "T": mo.ui.slider(1.0, 4.0, step=0.05, value=2.27, label="Temperature T", debounce=True),
        "n_sweeps": mo.ui.slider(100, 2000, step=100, value=800, label="Number of sweeps", debounce=True),
        "frames_every": mo.ui.slider(1, 20, value=4, label="Frame interval", debounce=True),
        "seed": mo.ui.number(0, label="Random seed", value=42),
    })
    controls
    return (controls,)


@app.cell
def _(controls):
    L = controls.value["L"]
    T = controls.value["T"]
    n_sweeps = controls.value["n_sweeps"]
    frames_every = controls.value["frames_every"]
    seed = controls.value["seed"]
    return L, T, frames_every, n_sweeps, seed


@app.cell
def _(L, T, frames_every, n_sweeps, run_ising, seed):
    simulation = run_ising(L, T, n_sweeps, frames_every, seed)
    return (simulation,)


@app.cell
def _(plt, simulation):
    mag = simulation["mag"]
    # Magnetization plot
    fig2, ax2 = plt.subplots()
    ax2.plot(mag)
    ax2.set_xlabel("Sweep")
    ax2.set_ylabel("Magnetisation per spin")
    ax2.set_title("Magnetisation over time")
    ax2
    return


@app.cell
def _(mo, simulation):
    frames = simulation["frames"]

    time = mo.ui.slider(0, len(frames)-1, 1, label="FrameNo")
    time
    return frames, time


@app.cell
def _(frames, plt, time):
    # Final configuration
    fig1, ax1 = plt.subplots()
    ax1.imshow(frames[time.value], cmap="gray", interpolation="nearest")
    ax1.set_title(f"Lattice at frame {time.value}")
    ax1.axis("off")
    ax1
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
