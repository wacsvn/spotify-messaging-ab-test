"""
Tiered simulation for the Smart Queue A/B test.

Design targets (pre-registered, do NOT tune these to results):
  - Baseline OEC: 840 min / enrolled user / week, between-user SD ~360
  - Compliance: 40% of Treatment users ever use the banner (Control: 0%)
  - Assumed TRUE effect: ~+2.8% ITT (just above the 2% MDE), concentrated in
    compliers, with a novelty bump in weeks 1-2 that settles by week 3
  - Cannibalization: banner minutes partly replace autoplay; search unchanged
  - W4 retention 95.2% / 94.8%, crash rate 0.20% / 0.22% (control / treatment)

Structure: assignment -> compliance -> per-user baseline multiplier ->
daily noise -> treatment effect (compliers only) -> churn -> crashes.
"""

import math
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config ---
SEED = 42
N_PER_VARIANT = 7210
DAYS = 28
COMPLIANCE = 0.40

# Baseline daily behavior (minutes). 30 + 90 = 120 min/day = 840 min/week.
SEARCH_MEAN, SEARCH_SD = 30.0, 15.0
AUTOPLAY_MEAN, AUTOPLAY_SD = 90.0, 45.0
MINUTES_PER_SEARCH_SESSION = 12.5  # 30 min/day -> 2.4 sessions/day -> 16.8/week

# Between-user heterogeneity: lognormal multiplier with mean 1.
# Tuned so the control weekly SD lands near the pre-registered 360.
SIGMA_USER = 0.40

# Treatment effect for compliers, steady state (minutes/day, before the
# per-user multiplier and novelty profile). Net = +7.5 min/day.
SMART_QUEUE_MEAN, SMART_QUEUE_SD = 11.0, 4.0
CANNIBALIZE_AUTOPLAY_MEAN, CANNIBALIZE_AUTOPLAY_SD = 3.5, 1.5


# Novelty: effect is stronger early and settles by week 3.
def novelty_multiplier(day: int) -> float:
    if day <= 7:
        return 1.40
    if day <= 14:
        return 1.15
    return 1.00


RETENTION_P = {"control": 0.952, "treatment": 0.948}
CRASH_P = {"control": 0.0020, "treatment": 0.0022}


def _clipped_normal_mean(mu: float, sd: float) -> float:
    """E[max(0, N(mu, sd))] so the calibration accounts for clamping at 0."""
    z = mu / sd
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return mu * cdf + sd * pdf


def main():
    rng = np.random.default_rng(SEED)
    n_total = N_PER_VARIANT * 2

    # 1. Assignment: independent Bernoulli trials (realistic; makes SRM check meaningful)
    ids = rng.choice(16**8, size=n_total, replace=False)  # unique 8-hex ids, seeded
    user_ids = np.array([f"{i:08x}" for i in ids])
    variant = rng.choice(["control", "treatment"], size=n_total, p=[0.5, 0.5])
    is_treat = variant == "treatment"

    # 2. Compliance: exactly 40% of the realized treatment arm
    complier = np.zeros(n_total, dtype=int)
    treat_idx = np.flatnonzero(is_treat)
    chosen = rng.choice(treat_idx, size=int(len(treat_idx) * COMPLIANCE), replace=False)
    complier[chosen] = 1
    is_complier_treated = (is_treat & (complier == 1))[:, None]

    # 3. Per-user baseline multiplier (mean 1) -> between-user variance
    mult = rng.lognormal(mean=-0.5 * SIGMA_USER**2, sigma=SIGMA_USER, size=n_total)[
        :, None
    ]

    # 4. Calibrate base daily means so enrolled-user mean ~ 840 min/week
    #    after clamping and W4 churn (days 21-28 missing for churned users).
    raw_daily = _clipped_normal_mean(SEARCH_MEAN, SEARCH_SD) + _clipped_normal_mean(
        AUTOPLAY_MEAN, AUTOPLAY_SD
    )
    present_frac = 1 - (1 - RETENTION_P["control"]) * 8 / DAYS
    cal = 840.0 / (7 * raw_daily * present_frac)

    shape = (n_total, DAYS)
    search = np.clip(rng.normal(SEARCH_MEAN * cal, SEARCH_SD, shape), 0, None) * mult
    autoplay = (
        np.clip(rng.normal(AUTOPLAY_MEAN * cal, AUTOPLAY_SD, shape), 0, None) * mult
    )

    # 5. Treatment effect (compliers only), scaled by user multiplier and novelty
    nov = np.array([novelty_multiplier(d) for d in range(1, DAYS + 1)])[None, :]
    sq_raw = np.clip(rng.normal(SMART_QUEUE_MEAN, SMART_QUEUE_SD, shape), 0, None)
    cann_raw = np.clip(
        rng.normal(CANNIBALIZE_AUTOPLAY_MEAN, CANNIBALIZE_AUTOPLAY_SD, shape), 0, None
    )
    smart_queue = np.where(is_complier_treated, sq_raw * mult * nov, 0.0)
    autoplay = np.where(
        is_complier_treated,
        np.clip(autoplay - cann_raw * mult * nov, 0, None),
        autoplay,
    )

    total = search + autoplay + smart_queue
    search_sessions = rng.poisson(search / MINUTES_PER_SEARCH_SESSION)

    # 6. Retention (W4 churn) and crashes
    p_ret = np.where(is_treat, RETENTION_P["treatment"], RETENTION_P["control"])
    retained = rng.random(n_total) < p_ret
    p_crash = np.where(is_treat, CRASH_P["treatment"], CRASH_P["control"])[:, None]
    crashes = rng.binomial(1, p_crash, shape)

    day_grid = np.tile(np.arange(1, DAYS + 1), (n_total, 1))
    present = ~((day_grid >= 21) & (~retained[:, None]))  # churned users vanish in W4

    df = pd.DataFrame(
        {
            "user_id": np.repeat(user_ids, DAYS)[present.ravel()],
            "variant": np.repeat(variant, DAYS)[present.ravel()],
            "is_complier": np.repeat(complier, DAYS)[present.ravel()],
            "experiment_day": day_grid.ravel()[present.ravel()],
            "search_minutes": search.ravel()[present.ravel()],
            "search_sessions": search_sessions.ravel()[present.ravel()],
            "autoplay_minutes": autoplay.ravel()[present.ravel()],
            "smart_queue_minutes": smart_queue.ravel()[present.ravel()],
            "total_minutes": total.ravel()[present.ravel()],
            "crashes": crashes.ravel()[present.ravel()],
            "w4_active_day": (day_grid.ravel()[present.ravel()] >= 21).astype(int),
        }
    )

    # 7. Sanity report vs. design targets
    print(f"Rows: {len(df):,} | Users: {df.user_id.nunique():,}")
    print(df.groupby("variant").user_id.nunique().to_string())
    per_user = df.groupby(["user_id", "variant", "is_complier"], as_index=False).agg(
        total=("total_minutes", "sum"), sessions=("search_sessions", "sum")
    )
    per_user["weekly"] = per_user.total / 4
    stats = per_user.groupby("variant").weekly.agg(["mean", "std"])
    print("\nWeekly minutes / enrolled user (target mean 840, SD ~360):")
    print(stats.round(1).to_string())
    c = per_user[per_user.variant == "control"].weekly
    t = per_user[per_user.variant == "treatment"].weekly
    itt = t.mean() - c.mean()
    print(
        f"\nITT lift: {itt:+.1f} min/wk ({itt / c.mean() * 100:+.2f}%)  [MDE = +16.8, 2.0%]"
    )
    comp = per_user[
        (per_user.variant == "treatment") & (per_user.is_complier == 1)
    ].weekly
    noncomp = per_user[
        (per_user.variant == "treatment") & (per_user.is_complier == 0)
    ].weekly
    print(
        f"Complier lift vs control: {comp.mean() - c.mean():+.1f} | Non-complier vs control: {noncomp.mean() - c.mean():+.1f}"
    )

    df["week"] = (df.experiment_day - 1) // 7 + 1
    wk = df.groupby(["week", "variant"]).total_minutes.sum().unstack()
    n_by = df.groupby("variant").user_id.nunique()
    wk_pu = wk.div(n_by, axis=1)
    print("\nWeekly lift (treatment - control, per enrolled user) -> novelty decay:")
    print((wk_pu.treatment - wk_pu.control).round(1).to_string())

    print("\nW4 retention by variant:")
    print(
        df[df.week == 4]
        .groupby("variant")
        .user_id.nunique()
        .div(n_by)
        .round(4)
        .to_string()
    )
    print("Crash rate per user-day:")
    print(df.groupby("variant").crashes.mean().round(5).to_string())
    print("Search sessions / user / week:")
    print((df.groupby("variant").search_sessions.sum() / n_by / 4).round(2).to_string())
    print("Autoplay minutes / user / week:")
    print(
        (df.groupby("variant").autoplay_minutes.sum() / n_by / 4).round(1).to_string()
    )

    df = df.drop(columns="week")
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulated_experiment_log.csv"
    )
    df.to_csv(out, index=False)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
