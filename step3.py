import csv
import numpy as np
import math

# --- КОНСТАНТЫ ---
GM_SUN = 1.32712440018e11 

# --- ФУНКЦИИ (Stumpff & Propagate) ---
# (Копируем те же функции, что и в Шаге 2, чтобы скрипт был автономным)

def stumpff_S(z):
    if z > 0:
        sqz = np.sqrt(z)
        return (sqz - np.sin(sqz)) / (sqz**3)
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.sinh(sqz) - sqz) / ((-z)**1.5)
    return 1/6

def stumpff_C(z):
    if z > 0:
        sqz = np.sqrt(z)
        return (1 - np.cos(sqz)) / z
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.cosh(sqz) - 1) / (-z)
    return 1/2

def propagate_state(r0, v0, dt, mu):
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    vr0 = np.dot(r0, v0) / r0_mag
    alpha = (2 / r0_mag) - (v0_mag**2 / mu)
    
    # Начальное приближение chi
    chi = np.sqrt(mu) * dt * alpha if abs(alpha) > 1e-6 else 0
    
    ratio = 1.0
    iter_count = 0
    while abs(ratio) > 1e-7 and iter_count < 100:
        z = alpha * chi**2
        s_val = stumpff_S(z)
        c_val = stumpff_C(z)
        
        t_calc = (r0_mag * vr0 / np.sqrt(mu)) * (chi**2 * c_val) + \
                 (1 - alpha * r0_mag) * (chi**3 * s_val) + r0_mag * chi
        
        r_curr = r0_mag + (r0_mag * vr0 / np.sqrt(mu)) * chi * (1 - z * s_val) + \
                 (1 - alpha * r0_mag) * chi**2 * c_val
                 
        step = (dt * np.sqrt(mu) - t_calc) / r_curr
        chi += step
        ratio = step
        iter_count += 1

    z = alpha * chi**2
    f = 1 - (chi**2 / r0_mag) * stumpff_C(z)
    g = dt - (chi**3 / np.sqrt(mu)) * stumpff_S(z)
    
    return f * r0 + g * v0

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ LSQ ---

def get_ra_dec_from_vector(r_vec):
    """Превращает вектор (x,y,z) в углы RA, Dec (радианы)"""
    # r_vec - это топоцентрический вектор (от Земли к объекту)
    dist = np.linalg.norm(r_vec)
    l = r_vec[0] / dist # direction cosines
    m = r_vec[1] / dist
    n = r_vec[2] / dist
    
    dec = np.arcsin(n)
    ra = np.arctan2(m, l)
    
    # Нормализация RA в 0..2pi
    if ra < 0: ra += 2*np.pi
    
    return ra, dec

def compute_residuals_and_partials(state_vec, observations, t_epoch):
    """
    Считает разницу (O-C) и строит матрицу Якоби (производные)
    state_vec: [x, y, z, vx, vy, vz]
    observations: список словарей с данными
    """
    r0 = state_vec[:3]
    v0 = state_vec[3:]
    
    b_vec = [] # Вектор невязок (Residuals)
    A_mat = [] # Матрица частных производных (Jacobian)
    
    # Шаги для численной производной (Finite Differencing)
    # Сдвигаем позицию на 10 км, скорость на 10 м/с (0.01 км/с)
    perturbations = [10.0, 10.0, 10.0, 0.01, 0.01, 0.01] 
    
    for obs in observations:
        t_obs = obs['t']
        R_earth = obs['R_earth']
        ra_obs = obs['ra']   # радианы
        dec_obs = obs['dec'] # радианы
        
        dt = t_obs - t_epoch
        
        # 1. Номинальная траектория
        r_sat = propagate_state(r0, v0, dt, GM_SUN)
        rho_vec = r_sat - R_earth # Топоцентрический вектор
        
        ra_calc, dec_calc = get_ra_dec_from_vector(rho_vec)
        
        # Считаем невязки (Observed - Computed)
        d_ra = ra_obs - ra_calc
        d_dec = dec_obs - dec_calc
        
        # Обработка перехода через 0/360 для RA
        if d_ra > np.pi: d_ra -= 2*np.pi
        if d_ra < -np.pi: d_ra += 2*np.pi
        
        # В матрицу b добавляем две строки: невязка по RA и по Dec
        # Важно: RA обычно умножают на cos(Dec) для приведения к дуге
        b_vec.append(d_ra) 
        b_vec.append(d_dec)
        
        # 2. Численные производные (строим матрицу A)
        # Нам нужно понять, как dRA/dx, dDec/dx и т.д. меняются
        
        row_ra = []
        row_dec = []
        
        for i in range(6):
            # Создаем возмущенный вектор состояния
            perturbed_state = state_vec.copy()
            perturbed_state[i] += perturbations[i]
            
            r_p = perturbed_state[:3]
            v_p = perturbed_state[3:]
            
            # Пропагируем
            r_sat_p = propagate_state(r_p, v_p, dt, GM_SUN)
            rho_vec_p = r_sat_p - R_earth
            
            ra_p, dec_p = get_ra_dec_from_vector(rho_vec_p)
            
            # Производная = (Val_perturbed - Val_nominal) / perturbation
            d_ra_val = (ra_p - ra_calc)
            if d_ra_val > np.pi: d_ra_val -= 2*np.pi
            if d_ra_val < -np.pi: d_ra_val += 2*np.pi
            
            deriv_ra = d_ra_val / perturbations[i]
            deriv_dec = (dec_p - dec_calc) / perturbations[i]
            
            row_ra.append(deriv_ra)
            row_dec.append(deriv_dec)
            
        A_mat.append(row_ra)
        A_mat.append(row_dec)
        
    return np.array(A_mat), np.array(b_vec)

# --- ЗАГРУЗКА ДАННЫХ ---

print("=== ШАГ 3: ДИФФЕРЕНЦИАЛЬНАЯ КОРРЕКЦИЯ (LSQ) ===")

# 1. Читаем начальное приближение (из Гаусса)
try:
    with open("data.txt", "r") as f:
        lines = f.readlines()
        t_epoch = float(lines[0].strip())
        r_epoch = np.array([float(x) for x in lines[1].split()])
        v_epoch = np.array([float(x) for x in lines[2].split()])
        current_state = np.concatenate((r_epoch, v_epoch))
        print("Начальное состояние загружено.")
except:
    print("Ошибка: Сначала выполните Шаг 1 (Метод Гаусса)!")
    exit()

# 2. Читаем все наблюдения
observations = []
with open('data.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)
    for row in reader:
        if len(row) < 9: continue
        t = float(row[1])
        # Векторы из файла нам нужны для перевода обратно в углы (если их там нет)
        # Но у нас в CSV есть x, y, z (direction cosine). Переведем их обратно в RA/Dec
        # x = cos d cos a, y = cos d sin a, z = sin d
        x, y, z = float(row[2]), float(row[3]), float(row[4])
        R_earth = np.array([float(row[6]), float(row[7]), float(row[8])])
        
        dec_rad = np.arcsin(z)
        ra_rad = np.arctan2(y, x)
        if ra_rad < 0: ra_rad += 2*np.pi
        
        observations.append({
            't': t,
            'ra': ra_rad,
            'dec': dec_rad,
            'R_earth': R_earth
        })

print(f"Загружено {len(observations)} наблюдений.")

# --- ЦИКЛ КОРРЕКЦИИ ---

MAX_ITER = 10
RMS_history = []

for it in range(MAX_ITER):
    print(f"\nИтерация {it+1}...")
    
    # Строим матрицы
    # A * delta_x = b
    try:
        A, b = compute_residuals_and_partials(current_state, observations, t_epoch)
    except Exception as e:
        print(f"Ошибка при расчете матриц: {e}")
        break
    
    # Считаем текущую ошибку (RMS) в угловых секундах
    # b содержит радианы. 
    rms_rad = np.sqrt(np.mean(b**2))
    rms_arcsec = np.degrees(rms_rad) * 3600.0
    RMS_history.append(rms_arcsec)
    
    print(f"  Текущая RMS ошибка: {rms_arcsec:.4f} arcsec")
    
    if it > 0 and abs(RMS_history[-2] - RMS_history[-1]) < 0.001:
        print("  Сходимость достигнута (ошибка не меняется).")
        break
        
    # Решаем систему нормальных уравнений: (A.T * A) * delta = A.T * b
    # Или используем lstsq (более устойчив)
    delta_state, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Поправка к состоянию
    # ВНИМАНИЕ: b = Obs - Calc. Значит A * delta = Obs - Calc
    # Мы прибавляем delta к текущему состоянию
    current_state += delta_state
    
    print(f"  Поправка положения: {np.linalg.norm(delta_state[:3]):.3f} км")
    print(f"  Поправка скорости:  {np.linalg.norm(delta_state[3:]):.5f} км/с")

print("\n" + "="*30)
print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
print("="*30)
print(f"Epoch (t2): {t_epoch}")
print(f"Pos (r): {current_state[:3]}")
print(f"Vel (v): {current_state[3:]}")

# Сохраняем улучшенное решение
with open("data1.txt", "w") as f:
    f.write(f"{t_epoch}\n")
    f.write(f"{current_state[0]} {current_state[1]} {current_state[2]}\n")
    f.write(f"{current_state[3]} {current_state[4]} {current_state[5]}\n")
print("Сохранено в 'orbit_solution_refined.txt'. Теперь запустите Шаг 2 для проверки этого файла!")