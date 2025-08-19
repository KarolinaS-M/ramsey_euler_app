# Ramsey Model (Euler) — Streamlit interactive simulator
# All comments are in English, as requested.

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Set Streamlit page configuration (wide layout for side-by-side plots)
st.set_page_config(page_title="Ramsey (Euler) Simulator", layout="wide")

# ---------- Model building blocks ----------
def f(k: float, alpha: float) -> float:
    """Cobb–Douglas production per capita: y = k^alpha."""
    return k ** alpha

def fprime(k: float, alpha: float) -> float:
    """Marginal product of capital: f'(k) = alpha * k^(alpha-1)."""
    # Guard at k = 0 to avoid k**(alpha-1) blowing up when alpha < 1
    return alpha * (k ** (alpha - 1.0)) if k > 0 else 0.0

def rhs(k: float, c: float, alpha: float, delta: float, n: float, rho: float, theta: float):
    """
    Right-hand side of the continuous-time Ramsey system with CRRA utility.
    dk/dt = f(k) - c - (n + delta) k
    dc/dt = (c/theta) * (f'(k) - delta - rho)
    """
    dk = f(k, alpha) - c - (n + delta) * k
    dc = (c / theta) * (fprime(k, alpha) - delta - rho)
    return dk, dc

def euler_path(k0: float, c0: float, h: float, T: float,
               alpha: float, delta: float, n: float, rho: float, theta: float):
    """
    Explicit (forward) Euler discretization over [0, T] with step h.
    Returns time grid and simulated paths for k(t) and c(t).
    """
    # Build uniform time grid
    tgrid = np.arange(0.0, T + 1e-12, h)
    N = len(tgrid)

    # Preallocate arrays
    k = np.empty(N)
    c = np.empty(N)

    # Initial conditions
    k[0], c[0] = k0, c0

    # Time stepping
    for i in range(N - 1):
        dk, dc = rhs(k[i], c[i], alpha, delta, n, rho, theta)
        k_next = k[i] + h * dk
        c_next = c[i] + h * dc

        # Numerical guards: no negative capital; tiny floor for consumption
        k[i + 1] = max(k_next, 0.0)
        c[i + 1] = max(c_next, 1e-300)  # avoid division-by-zero in TVC

    return tgrid, k, c

def steady_state(alpha: float, rho: float, delta: float, n: float):
    """
    Closed-form steady state for the Ramsey model (with CRRA and Cobb–Douglas).
    Uses the golden-rule-type condition f'(k*) = rho + delta.
    """
    k_star = (alpha / (rho + delta)) ** (1.0 / (1.0 - alpha))
    c_star = f(k_star, alpha) - (n + delta) * k_star
    return k_star, c_star

def tvc_series(k: np.ndarray, c: np.ndarray, tgrid: np.ndarray, n: float, rho: float, theta: float):
    """
    Transversality condition diagnostic series:
    TVC(t) = c(t)^(-theta) * k(t) * exp((n - rho) * t).
    In continuous time, TVC -> 0 should hold along the optimal path.
    """
    return (c ** (-theta)) * k * np.exp((n - rho) * tgrid)

# ---------- Sidebar controls ----------
st.sidebar.header("Parameters")
alpha = st.sidebar.slider("α (Cobb–Douglas exponent)", 0.05, 0.95, 0.33, 0.01)
delta = st.sidebar.slider("δ (depreciation)", 0.00, 0.20, 0.05, 0.005)
n     = st.sidebar.slider("n (population growth)", -0.05, 0.10, 0.01, 0.001)
rho   = st.sidebar.slider("ρ (discount rate)", 0.00, 0.20, 0.02, 0.005)
theta = st.sidebar.slider("θ (inverse IES)", 0.10, 5.00, 2.00, 0.10)

st.sidebar.header("Numerics")
h = st.sidebar.slider("Euler step h", 0.001, 1.000, 0.100, 0.001)
T = st.sidebar.slider("Horizon T", 10.0, 5000.0, 2200.0, 10.0)

st.sidebar.header("Initial conditions")
k0 = st.sidebar.number_input("k(0)", value=10.0, step=0.5)
c0 = st.sidebar.number_input("c(0)", value=1.0, step=0.1)

st.sidebar.header("Plot options")
tvc_scale_log = st.sidebar.checkbox("Log scale for TVC", value=False)
show_csv      = st.sidebar.checkbox("Show data table preview", value=False)

# ---------- Compute simulation ----------
k_star, c_star = steady_state(alpha, rho, delta, n)
t, k, c = euler_path(k0, c0, h, T, alpha, delta, n, rho, theta)
tvc = tvc_series(k, c, t, n, rho, theta)

# ---------- Page header ----------
st.title("Ramsey Model (Euler) — Interactive Simulator")

st.markdown(
    f"""
**Steady state**:

- \(k^* = {k_star:.5f}\)  
- \(c^* = {c_star:.5f}\)  

**TVC at T**: \( {tvc[-1]:.3e} \)
"""
)

# ---------- Plots: k(t) and c(t) side-by-side ----------
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(t, k, label="k(t)")
    ax1.axhline(k_star, ls="--", alpha=0.6, label="k*")
    ax1.set_title("Capital per capita k(t)")
    ax1.set_xlabel("t")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(t, c, label="c(t)")
    ax2.axhline(c_star, ls="--", alpha=0.6, label="c*")
    ax2.set_title("Consumption per capita c(t)")
    ax2.set_xlabel("t")
    ax2.legend()
    st.pyplot(fig2)

# ---------- Separate TVC figure ----------
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(t, tvc, label="TVC(t)")
ax3.set_title("Transversality condition")
ax3.set_xlabel("t")
if tvc_scale_log:
    ax3.set_yscale("log")
ax3.legend()
st.pyplot(fig3)

# ---------- Optional table preview and CSV download ----------
if show_csv:
    df = pd.DataFrame({"t": t, "k(t)": k, "c(t)": c, "TVC": tvc})
    st.dataframe(df.head(200))
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="ramsey_euler.csv",
        mime="text/csv",
    )

# ---------- Footnote with discretization ----------
st.caption(
    "Euler discretization: "
    r"$k_{t+h}=k_t+h\,[k_t^{\alpha}-c_t-(n+\delta)k_t]$, "
    r"$c_{t+h}=c_t+h\,\big[\frac{c_t}{\theta}(\alpha k_t^{\alpha-1}-\delta-\rho)\big]$. "
    "Note: For a saddle-path system, explicit Euler can diverge unless the initial point lies on the stable manifold."
)
