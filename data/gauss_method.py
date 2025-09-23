#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orbit_refine_lsq_fixed.py

Исправленная версия твоего скрипта:
 - избегает дорогих/зацикливающих операций (minimize_scalar),
 - использует izzo для начального приближения по 3 наблюдениям,
 - исправлены баги с propagate_universal и read_and_prepare,
 - добавлены защиты в LM (patience, lambda cap).
Комментарии на русском.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, get_body_barycentric, get_body_barycentric_posvel
import astropy.units as u
from astropy import constants as const
from math import acos, atan2, sqrt
import logging

# poliastro (нужен)
from poliastro.bodies import Sun
from poliastro.iod import izzo
from poliastro.twobody import Orbit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Константы ----
AU = u.au
DAY = u.day
C = const.c.to(u.au / u.day).value  # скорость света в AU/day
MU = const.GM_sun.to((AU**3) / (DAY**2)).value  # ГМ Солнца (AU^3/day^2)

# ---- Вспомогательные функции ----
def norm(v: np.ndarray) -> float:
    return np.linalg.norm(v)

def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    return v / n if n > 0 else v

# ---- Stumpff (как в твоём коде) ----
def stumpff_C(z: float) -> float:
    if z > 1e-4:
        sz = sqrt(z)
        return (1.0 - np.cos(sz)) / z
    elif z < -1e-4:
        sz = sqrt(-z)
        return (1.0 - np.cosh(sz)) / z if abs(z) < 700 else 0.5
    else:
        return 0.5 - z / 24.0 + z**2 / 720.0

def stumpff_S(z: float) -> float:
    if z > 1e-4:
        sz = sqrt(z)
        return (sz - np.sin(sz)) / (sz**3)
    elif z < -1e-4:
        sz = sqrt(-z)
        return (np.sinh(sz) - sz) / (sz**3) if abs(z) < 700 else 1.0/6.0
    else:
        return 1.0/6.0 - z/120.0 + z**2/5040.0

# ---- Propagate universal (устойчивее и с корректным возвратом) ----
def propagate_universal(r0: np.ndarray, v0: np.ndarray, dt_days: float, mu: float = MU):
    r0 = np.array(r0, dtype=float)
    v0 = np.array(v0, dtype=float)
    r0_norm = norm(r0)
    v0_norm = norm(v0)
    if r0_norm == 0:
        return r0, v0
    vr0 = np.dot(r0, v0) / r0_norm
    alpha = 2.0 / r0_norm - v0_norm**2 / mu

    # initial guess for chi
    if abs(alpha) < 1e-8:
        chi = sqrt(mu) * abs(dt_days) / max(r0_norm, 1e-6)
    elif alpha > 0:
        chi = sqrt(mu) * dt_days * alpha
    else:
        a = 1.0 / alpha if abs(alpha) > 1e-8 else -1.0
        sign_dt = np.sign(dt_days) if dt_days != 0 else 1.0
        denom = vr0 + sign_dt * sqrt(-mu * a) * (1.0 - r0_norm * alpha)
        arg = (-2.0 * mu * alpha * dt_days) / denom if denom != 0 else -1.0
        if arg > 0:
            chi = sign_dt * sqrt(-a) * np.log(max(arg, 1e-12))
        else:
            chi = sqrt(mu) * dt_days / max(r0_norm, 1e-6)

    max_iter = 60
    tol = 1e-11
    for _ in range(max_iter):
        z = alpha * chi**2
        C = stumpff_C(z)
        S = stumpff_S(z)
        F = (r0_norm * chi * (1.0 - z * S) + vr0 * chi**2 * C + sqrt(mu) * chi**3 * S) / sqrt(mu) - dt_days
        dF = (r0_norm * (1.0 - z * S) + vr0 * chi * C + sqrt(mu) * chi**2 * S) / sqrt(mu)
        if not np.isfinite(dF) or abs(dF) < 1e-16:
            # fallback finite difference derivative
            h = 1e-6 * max(1.0, abs(chi))
            chi_h = chi + h
            Fh = (r0_norm * chi_h * (1.0 - alpha * chi_h**2 * stumpff_S(alpha * chi_h**2)) +
                  vr0 * chi_h**2 * stumpff_C(alpha * chi_h**2) +
                  sqrt(mu) * chi_h**3 * stumpff_S(alpha * chi_h**2)) / sqrt(mu) - dt_days
            dF = (Fh - F) / h if np.isfinite(Fh) else 1.0
        delta = F / dF
        chi -= delta
        if not np.isfinite(chi):
            chi = sqrt(mu) * dt_days / max(r0_norm, 1e-6)
            break
        if abs(delta) < tol:
            break

    z = alpha * chi**2
    C = stumpff_C(z); S = stumpff_S(z)
    f = 1.0 - (chi**2 / r0_norm) * C
    g = dt_days - (chi**3 / sqrt(mu)) * S
    r = f * r0 + g * v0
    r_norm = norm(r)
    if r_norm == 0 or r0_norm == 0:
        return r0, v0
    fdot = (sqrt(mu) / (r_norm * r0_norm)) * (alpha * chi**3 * S - chi)
    gdot = 1.0 - (chi**2 / r_norm) * C
    v = fdot * r0 + gdot * v0

    if np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
        return r, v
    else:
        return r0, v0

# ---- Чтение наблюдений (упрощённо, без тяжёлой итерации светового времени) ----
def read_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path, comment='#')
    required = ['JD', 'RA_deg', 'Dec_deg', 'obs_lat_deg', 'obs_lon_deg', 'obs_alt_m']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSV missing required column {r}")
    times = Time(df['JD'].to_numpy(), format='jd', scale='utc')
    N = len(df)
    los_obs = np.zeros((N, 3))
    R_obs = np.zeros((N, 3))
    weights = None
    if 'RA_err_arcsec' in df.columns and 'Dec_err_arcsec' in df.columns:
        sigma2 = (df['RA_err_arcsec']**2 + df['Dec_err_arcsec']**2) * (np.pi/180/3600)**2
        weights = 1.0 / sigma2.to_numpy()
    for i, row in df.iterrows():
        t = times[i]
        loc = EarthLocation(lat=row['obs_lat_deg']*u.deg,
                            lon=row['obs_lon_deg']*u.deg,
                            height=row['obs_alt_m']*u.m)
        sc = SkyCoord(ra=row['RA_deg']*u.deg, dec=row['Dec_deg']*u.deg,
                      frame='icrs', obstime=t, location=loc, distance=1*AU)
        cart = sc.cartesian.xyz.to(AU).value
        los_obs[i, :] = unit(cart)
        earth_bary = get_body_barycentric('earth', t).xyz.to(AU).value.flatten()
        try:
            itrs = loc.get_itrs(obstime=t)
            icrs = itrs.transform_to('icrs')
            loc_icrs = icrs.cartesian.xyz.to(AU).value.flatten()
        except Exception:
            loc_icrs = np.array([loc.x.to(u.m).value, loc.y.to(u.m).value, loc.z.to(u.m).value]) * u.m.to(AU)
        R_obs[i, :] = earth_bary + loc_icrs
    return times, los_obs, R_obs, weights, df

# ---- Начальное приближение: сначала izzo по 3 наблюдениям, иначе простой fallback ----
def get_initial_state(t0: Time, times: Time, los_obs: np.ndarray, R_obs: np.ndarray):
    N = len(times)
    i1, i2, i3 = 0, N//2, N-1
    try:
        # izzo хочет R_obs в км и rho_hat как массив 3x3
        R_km = (R_obs[[i1,i2,i3]] * u.au).to(u.km)
        rhohats = los_obs[[i1,i2,i3]]
        t3 = times[[i1,i2,i3]]
        r2_km, v2_km = izzo.iod_izzo(Sun.k, R_km, rhohats, t3)
        r2 = r2_km.to(u.au).value
        v2 = v2_km.to(u.au/u.day).value
        return r2, v2
    except Exception as e:
        logger.warning("izzo failed: %s. Using fallback JPL-like initial guess (no expensive optimization).", e)
        # Fallback: JPL-like constant values (быстрый, детерминированный)
        a = -1.87 * u.au
        e = 1.199 * u.dimensionless_unscaled
        inc = 122.74 * u.deg
        raan = 24.6 * u.deg
        argp = 241.8 * u.deg
        nu = 0.0 * u.deg
        orb = Orbit.from_classical(Sun, a, e, inc, raan, argp, nu, epoch=t0)
        return orb.r.to(u.au).value, orb.v.to(u.au/u.day).value

# ---- predict LOS используя propagate_universal ----
def predict_los_for_state(state0: np.ndarray, t0: Time, times: Time, R_obs: np.ndarray):
    r0, v0 = state0[:3], state0[3:]
    N = len(times)
    los_pred = np.zeros((N, 3))
    for i in range(N):
        dt = times[i].jd - t0.jd
        ri, _ = propagate_universal(r0, v0, dt)
        los_pred[i, :] = unit(ri - R_obs[i])
    return los_pred

# ---- Построение невязок и Якобиана (форвардные разности: быстрее, стабильнее для старта) ----
def build_residual_and_jacobian(state0: np.ndarray, t0: Time, times: Time,
                                los_obs: np.ndarray, R_obs: np.ndarray, weights: np.ndarray = None,
                                eps_r: float = 1e-5, eps_v: float = 1e-6):
    N = len(times)
    los0 = predict_los_for_state(state0, t0, times, R_obs)
    y = (los0 - los_obs).reshape(3*N)
    if weights is not None:
        w = np.repeat(weights, 3)
        y = y * np.sqrt(w)
    else:
        w = None
    J = np.zeros((3*N, 6))
    deltas = np.array([eps_r, eps_r, eps_r, eps_v, eps_v, eps_v], dtype=float)
    for j in range(6):
        dstate = np.zeros(6)
        dstate[j] = deltas[j]
        los_p = predict_los_for_state(state0 + dstate, t0, times, R_obs)
        deriv = (los_p - los0) / deltas[j]
        if w is not None:
            J[:, j] = deriv.reshape(3*N) * np.sqrt(w)
        else:
            J[:, j] = deriv.reshape(3*N)
        if not np.all(np.isfinite(J[:, j])):
            J[:, j] = np.zeros(3*N)
    return y, J

# ---- Levenberg-Marquardt с защитами (patience, lambda cap) ----
def lm_least_squares(initial_state: np.ndarray, t0: Time, times: Time,
                     los_obs: np.ndarray, R_obs: np.ndarray, weights: np.ndarray = None,
                     max_iter: int = 30, tol: float = 1e-9, lamb0: float = 1.0):
    state = initial_state.copy()
    lam = lamb0
    history = []
    no_improve = 0
    LAMB_MAX = 1e12
    for it in range(max_iter):
        y, J = build_residual_and_jacobian(state, t0, times, los_obs, R_obs, weights)
        N = len(times)
        if not np.all(np.isfinite(y)):
            logger.error("Non-finite residuals encountered; aborting LM loop.")
            break
        rms_rad = np.sqrt(np.mean(np.sum((y.reshape(N, 3))**2, axis=1)))
        rms_arcsec = rms_rad * 206264.806
        history.append(rms_arcsec)
        logger.info("Iter %d: RMS = %.6f arcsec, lambda = %.3e", it, rms_arcsec, lam)
        JTJ = J.T @ J
        g = J.T @ y
        A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-16)
        try:
            dx = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(A, -g, rcond=None)[0]
        state_new = state + dx
        y_new, _ = build_residual_and_jacobian(state_new, t0, times, los_obs, R_obs, weights)
        rms_new = np.inf
        if np.all(np.isfinite(y_new)):
            rms_new = np.sqrt(np.mean(np.sum((y_new.reshape(N, 3))**2, axis=1))) * 206264.806
        if rms_new < rms_arcsec:
            state = state_new
            lam = max(lam / 10.0, 1e-12)
            no_improve = 0
            logger.info("Accepted: RMS improved to %.6f arcsec, |dx|=%.3e", rms_new, norm(dx))
        else:
            lam *= 10.0
            no_improve += 1
            logger.info("Rejected: RMS_try=%.6f arcsec, increasing lambda to %.3e", rms_new, lam)
        if norm(dx) < tol:
            logger.info("Converged: small dx (norm=%.3e)", norm(dx))
            break
        if lam > LAMB_MAX:
            logger.warning("Lambda exceeded %g — stopping to avoid instability", LAMB_MAX)
            break
        if no_improve >= 6:
            logger.info("No improvement in %d consecutive steps — stopping", no_improve)
            break
    return state, history

# ---- Перевод в классические элементы (через poliastro) ----
def classical_elements_from_rv(r: np.ndarray, v: np.ndarray):
    rq = r * u.au
    vq = v * u.au / u.day
    orb = Orbit.from_vectors(Sun, rq, vq, epoch=Time.now())
    return {
        'a_AU': orb.a.to(u.AU).value,
        'e': orb.ecc.value,
        'i_deg': orb.inc.to(u.deg).value,
        'Omega_deg': orb.raan.to(u.deg).value,
        'omega_deg': orb.argp.to(u.deg).value,
        'true_anomaly_deg': orb.nu.to(u.deg).value
    }

# ---- Проверки данных (как у тебя) ----
def check_data(df, times, los_obs):
    logger.info("Checking data for anomalies...")
    for i in range(len(df)):
        ra, dec = df['RA_deg'][i], df['Dec_deg'][i]
        if not (0 <= ra <= 360 and -90 <= dec <= 90):
            logger.warning("Obs %d: invalid RA/Dec: %s / %s", i, ra, dec)
        if np.any(~np.isfinite(los_obs[i])):
            logger.warning("Obs %d: invalid LOS vector", i)
    if np.any(np.diff(times.jd) <= 0):
        logger.warning("Non-monotonic JD detected")

# ---- Main ----
def main(csv_path: str):
    times, los_obs, R_obs, weights, df = read_and_prepare(csv_path)
    N = len(times)
    logger.info("Loaded %d observations", N)
    check_data(df, times, los_obs)

    t0 = Time(times[N//2].jd, format='jd', scale='utc')
    r0, v0 = get_initial_state(t0, times, los_obs, R_obs)
    initial_state = np.hstack([r0, v0])
    logger.info("Initial state (r0 AU): %s", r0)
    logger.info("Initial state (v0 AU/day): %s", v0)

    los_pred0 = predict_los_for_state(initial_state, t0, times, R_obs)
    angles_rad = np.arccos(np.clip(np.sum(los_pred0 * los_obs, axis=1), -1.0, 1.0))
    rms0_arcsec = np.sqrt(np.mean(angles_rad**2)) * 206264.806 if np.all(np.isfinite(angles_rad)) else np.inf
    logger.info("Initial RMS angular residual: %.6f arcsec", rms0_arcsec)

    state_refined, hist = lm_least_squares(initial_state, t0, times, los_obs, R_obs, weights,
                                          max_iter=30, tol=1e-9, lamb0=1.0)

    r_ref = state_refined[:3]
    v_ref = state_refined[3:]
    logger.info("Refined r (AU): %s", r_ref)
    logger.info("Refined v (AU/day): %s", v_ref)
    elements = classical_elements_from_rv(r_ref, v_ref)
    print("\nRefined orbital elements:")
    for k, v in elements.items():
        print(f"{k}: {v:.8f}")

    los_final = predict_los_for_state(state_refined, t0, times, R_obs)
    ang_final = np.arccos(np.clip(np.sum(los_final * los_obs, axis=1), -1.0, 1.0))
    rms_final_arcsec = np.sqrt(np.mean(ang_final**2)) * 206264.806 if np.all(np.isfinite(ang_final)) else np.inf
    print(f"\nInitial RMS (arcsec) = {rms0_arcsec:.6f}")
    print(f"Final   RMS (arcsec) = {rms_final_arcsec:.6f}")

    print("\nObs residuals (arcsec):")
    for i in range(N):
        resid_arcsec = ang_final[i] * 206264.806 if np.isfinite(ang_final[i]) else np.nan
        print(f"{i:2d} JD={times[i].jd:.6f}  resid_arcsec = {resid_arcsec:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refine orbit by LSQ (fixed, safer)")
    parser.add_argument("--csv", default="data/omuamua_obs.csv", help="CSV with observations")
    args = parser.parse_args()
    main(args.csv)
