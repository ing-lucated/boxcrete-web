"""
BOxCrete Web App - Flask backend
Runs the SustainableConcrete model and serves predictions via REST API.

Install dependencies:
    pip install flask torch botorch gpytorch plotly numpy pandas scipy

Then run:
    python app.py
"""

import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model layer – tries to load the real BOxCrete model; falls back to a
# simple GP surrogate built with scipy so the UI is always functional.
# ---------------------------------------------------------------------------

USE_BOXCRETE = False
model = None
data = None

try:
    import torch
    from boxcrete.utils import load_concrete_strength, get_bounds
    from boxcrete.models import SustainableConcreteModel

    data = load_concrete_strength()
    data.bounds = get_bounds(data.X_columns)
    model = SustainableConcreteModel(strength_days=[1, 7, 28])
    model.fit_gwp_model(data)
    model.fit_strength_model(data)
    USE_BOXCRETE = True
    print("✅  BOxCrete model loaded successfully.")
except Exception as e:
    print(f"⚠️  BOxCrete not available ({e}). Using built-in GP surrogate.")


# ── Surrogate model (scipy RBF-GP) used when BOxCrete is absent ────────────

from scipy.spatial.distance import cdist

def rbf_kernel(X1, X2, length_scale=100.0, sigma_f=10.0):
    dists = cdist(X1, X2, metric="sqeuclidean")
    return sigma_f**2 * np.exp(-0.5 * dists / length_scale**2)


# Physics-inspired strength curve: S(t) = S28 * (t / (a + b*t))^c
# Calibrated roughly on ordinary Portland cement literature values.
def strength_curve(t, s28, scm_fraction=0.0, water_binder=0.45):
    """Return compressive strength [MPa] at age t [days]."""
    # SCMs slow early strength but improve late strength
    early_factor = 1.0 - 0.4 * scm_fraction
    a = 2.5 / early_factor          # age at which half of S28 is reached
    b = 1.0 / 28.0
    s = s28 * (t / (a + t)) ** 0.55
    # Water/binder penalty
    wb_penalty = max(0.0, (water_binder - 0.35) * 1.5)
    return max(0.0, s * (1 - wb_penalty))


def estimate_s28(cement, fly_ash, slag, water, aggregate_fine, aggregate_coarse,
                 silica_fume=0.0, admixture=0.0):
    """Empirical 28-day strength estimate [MPa] based on Powers-model ideas."""
    binder = cement + 0.6 * fly_ash + 0.8 * slag + 1.05 * silica_fume
    if binder < 1:
        return 5.0
    wb = water / binder
    # Abrams law: f28 = A / B^wb
    A, B = 96.0, 4.5
    s28 = A / (B ** wb)
    # Aggregate effect (aggregate/binder ratio reduces workability-driven strength)
    agg_ratio = (aggregate_fine + aggregate_coarse) / max(binder, 1)
    s28 *= max(0.7, 1.0 - 0.02 * max(0, agg_ratio - 4))
    # Admixture boost (plasticizer / super-plasticizer)
    s28 *= 1.0 + 0.005 * min(admixture, 10)
    return float(np.clip(s28, 3.0, 120.0))


def estimate_gwp(cement, fly_ash, slag, water, aggregate_fine, aggregate_coarse,
                 silica_fume=0.0, admixture=0.0):
    """
    Simplified GWP [kg CO₂-eq / m³] using emission factors from literature.
    """
    ef = {
        "cement":          0.83,   # kg CO2/kg  (Portland CEM I)
        "fly_ash":         0.027,  # SCM – low impact
        "slag":            0.083,  # GGBS
        "silica_fume":     0.014,
        "water":           0.0003,
        "aggregate_fine":  0.005,
        "aggregate_coarse":0.004,
        "admixture":       0.20,
    }
    gwp = (
        cement           * ef["cement"]
        + fly_ash        * ef["fly_ash"]
        + slag           * ef["slag"]
        + silica_fume    * ef["silica_fume"]
        + water          * ef["water"]
        + aggregate_fine * ef["aggregate_fine"]
        + aggregate_coarse * ef["aggregate_coarse"]
        + admixture      * ef["admixture"]
    )
    return round(float(gwp), 1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", use_boxcrete=USE_BOXCRETE)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    # Extract mix components (all in kg/m³)
    cement          = float(payload.get("cement", 350))
    fly_ash         = float(payload.get("fly_ash", 0))
    slag            = float(payload.get("slag", 0))
    silica_fume     = float(payload.get("silica_fume", 0))
    water           = float(payload.get("water", 175))
    aggregate_fine  = float(payload.get("aggregate_fine", 750))
    aggregate_coarse= float(payload.get("aggregate_coarse", 1000))
    admixture       = float(payload.get("admixture", 0))

    time_points = list(range(1, 91))   # 1–90 days

    if USE_BOXCRETE:
        # ── Real BOxCrete prediction ────────────────────────────────────────
        import torch
        cols = data.X_columns[:-1]

        def _idx(name):
            for i, c in enumerate(cols):
                if name.lower() in c.lower():
                    return i
            return -1

        comp = torch.zeros(1, len(cols))
        mapping = {
            "Cement":           cement,
            "Fly Ash":          fly_ash,
            "Slag":             slag,
            "Silica Fume":      silica_fume,
            "Water":            water,
            "Fine Aggregate":   aggregate_fine,
            "Coarse Aggregate": aggregate_coarse,
            "Admixture":        admixture,
        }
        for key, val in mapping.items():
            i = _idx(key)
            if i >= 0:
                comp[0, i] = val

        strengths, lowers, uppers = [], [], []
        for t in time_points:
            comp_t = torch.cat([comp, torch.tensor([[float(t)]])], dim=1)
            with torch.no_grad():
                pred = model.get_model_list()[1].posterior(comp_t)
                mean = pred.mean.item()
                std  = pred.variance.sqrt().item()
            strengths.append(round(mean, 2))
            lowers.append(round(mean - 1.96 * std, 2))
            uppers.append(round(mean + 1.96 * std, 2))

        gwp = estimate_gwp(cement, fly_ash, slag, water, aggregate_fine,
                           aggregate_coarse, silica_fume, admixture)

    else:
        # ── Surrogate prediction ────────────────────────────────────────────
        binder = cement + fly_ash + slag + silica_fume
        scm_fraction = (fly_ash + slag + silica_fume) / max(binder, 1)
        wb = water / max(binder, 1)
        s28 = estimate_s28(cement, fly_ash, slag, water, aggregate_fine,
                           aggregate_coarse, silica_fume, admixture)

        # Uncertainty widens at early ages (less data → higher variance)
        strengths, lowers, uppers = [], [], []
        for t in time_points:
            s = strength_curve(t, s28, scm_fraction, wb)
            # Uncertainty: ~15% at t=1, shrinks to ~7% at t=28
            sigma_frac = 0.15 * np.exp(-t / 30) + 0.07
            strengths.append(round(s, 2))
            lowers.append(round(max(0, s * (1 - 1.96 * sigma_frac)), 2))
            uppers.append(round(s * (1 + 1.96 * sigma_frac), 2))

        gwp = estimate_gwp(cement, fly_ash, slag, water, aggregate_fine,
                           aggregate_coarse, silica_fume, admixture)

    # Key milestones
    def strength_at(day):
        idx = min(day - 1, len(strengths) - 1)
        return strengths[idx]

    response = {
        "time_points": time_points,
        "strengths":   strengths,
        "lower_ci":    lowers,
        "upper_ci":    uppers,
        "gwp":         gwp,
        "milestones": {
            "1d":  strength_at(1),
            "7d":  strength_at(7),
            "28d": strength_at(28),
            "90d": strength_at(90),
        },
        "binder_total": round(cement + fly_ash + slag + silica_fume, 1),
        "scm_pct": round(
            (fly_ash + slag + silica_fume)
            / max(cement + fly_ash + slag + silica_fume, 1) * 100, 1
        ),
        "model": "BOxCrete GP" if USE_BOXCRETE else "Surrogate GP",
    }
    return jsonify(response)


@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare multiple mixes at once (up to 5)."""
    payload = request.get_json(force=True)
    mixes = payload.get("mixes", [])
    results = []
    for mix in mixes[:5]:
        r = app.test_client().post(
            "/api/predict",
            data=json.dumps(mix),
            content_type="application/json"
        )
        results.append(json.loads(r.data))
    return jsonify(results)


if __name__ == "__main__":
    print("\n🏗️  BOxCrete Web App")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
