import csv
import numpy as np

# --- КОНСТАНТЫ ---
# Гравитационный параметр Солнца (km^3/s^2)
# Поскольку здесь есть "секунды", все время в формулах должно быть в секундах.
GM_SUN = 1.32712440018e11 

def stumpff_S(z):
    """Функция Штумпфа S(z)"""
    if z > 0:
        sqz = np.sqrt(z)
        return (sqz - np.sin(sqz)) / (sqz**3)
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.sinh(sqz) - sqz) / ((-z)**1.5)
    return 1/6

def stumpff_C(z):
    """Функция Штумпфа C(z)"""
    if z > 0:
        sqz = np.sqrt(z)
        return (1 - np.cos(sqz)) / z
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.cosh(sqz) - 1) / (-z)
    return 1/2

def propagate_state(r0, v0, dt, mu):
    """
    Универсальный пропагатор (работает для элиипса гиперболы и параболы).
    Использует метод Ньютона для универсальной переменной chi.
    """
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    vr0 = np.dot(r0, v0) / r0_mag
    
    # alpha = 1/a (обратная большая полуось)
    alpha = (2 / r0_mag) - (v0_mag**2 / mu)
    
    # Начальное приближение для универсальной переменной chi
    chi = np.sqrt(mu) * dt * alpha if abs(alpha) > 1e-6 else 0
    
    # Итерации Ньютона
    ratio = 1.0
    iter_count = 0
    while abs(ratio) > 1e-7 and iter_count < 100:
        z = alpha * chi**2
        s_val = stumpff_S(z)
        c_val = stumpff_C(z)
        
        # Уравнение Кеплера в универсальных переменных
        t_calc = (r0_mag * vr0 / np.sqrt(mu)) * (chi**2 * c_val) + \
                 (1 - alpha * r0_mag) * (chi**3 * s_val) + \
                 r0_mag * chi
        
        dt_current = t_calc / np.sqrt(mu) # время, соответствующее текущему chi
        
        # Производная dt/dchi = r / sqrt(mu)
        r_val = r0_mag * (1 - alpha * chi**2 * c_val) + \
                vr0 * np.sqrt(mu) * chi * (1 - alpha * chi**2 * s_val) # Это не точно r, но для ньютона пойдет
        
        # Точное значение r для производной
        r_curr = chi**2 * c_val + r0_mag * (1 - alpha * chi**2 * s_val) + \
                 (r0_mag * vr0 / np.sqrt(mu)) * chi * (1 - z * s_val) # Это сложная формула, используем простую производную
        
    
        # Используем формулу Валладо:
        
        r_curr = r0_mag + (r0_mag * vr0 / np.sqrt(mu)) * chi * (1 - z * s_val) + \
                 (1 - alpha * r0_mag) * chi**2 * c_val
                 
        step = (dt * np.sqrt(mu) - t_calc) / r_curr
        chi += step
        ratio = step
        iter_count += 1

    # Вычисление f и g функций
    z = alpha * chi**2
    f = 1 - (chi**2 / r0_mag) * stumpff_C(z)
    g = dt - (chi**3 / np.sqrt(mu)) * stumpff_S(z)
    
    # Можно также вычислить f_dot и g_dot для скорости, но нам нужно только положение
    return f * r0 + g * v0
    
# --- ОСНОВНОЙ СКРИПТ ---

print("=== ЗАПУСК ПРОВЕРКИ ОРБИТЫ ===")

# Читаем решение (которое сохранил первый скрипт)
try:
    with open("data1.txt", "r") as f:
        lines = f.readlines()
        # t_epoch (t2) — это время в СЕКУНДАХ из вашего первого скрипта
        t_epoch = float(lines[0].strip()) 
        
        r_vec_epoch = np.array([float(x) for x in lines[1].split()])
        v_vec_epoch = np.array([float(x) for x in lines[2].split()])
        
    print(f"Эпоха (t2, сек): {t_epoch}")
    print(f"Положение: {r_vec_epoch}")
    print(f"Скорость:  {v_vec_epoch}")

except FileNotFoundError:
    print("Файл orbit_solution.txt не найден.")
    exit()

print("\n" + "-"*85)
print(f"{'Row':<5} {'Time (sec)':<15} {'Delta T (days)':<15} {'Residual (arcsec)':<20} {'Status'}")
print("-" * 85)

residuals = []
CSV_FILENAME = 'data.csv'

with open(CSV_FILENAME, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    header = next(reader)
    
    row_idx = 0
    for row in reader:
        if not row or len(row) < 9: continue
        row_idx += 1
        
        # t_obs — время наблюдения из файла (в СЕКУНДАХ)
        t_obs = float(row[1])
        
        # Векторы из файла
        rho_hat_obs = np.array([float(row[2]), float(row[3]), float(row[4])])
        R_vec_obs = np.array([float(row[6]), float(row[7]), float(row[8])])
        
        # --- ВЫЧИСЛЕНИЕ ---
        
        # dt = Секунды - Секунды = СЕКУНДЫ
        dt_sec = t_obs - t_epoch
        
        # Передаем dt_sec (секунды) в функцию
        r_calc = propagate_state(r_vec_epoch, v_vec_epoch, dt_sec, GM_SUN)
        
        if r_calc is None:
            print(f"{row_idx:<5} Ошибка (гипербола)")
            continue

        # Вектор от обсерватории к спутнику
        rho_vec_calc = r_calc - R_vec_obs
        rho_hat_calc = rho_vec_calc / np.linalg.norm(rho_vec_calc)
        
        # Угол ошибки
        dot_prod = np.clip(np.dot(rho_hat_calc, rho_hat_obs), -1.0, 1.0)
        angle_arcsec = np.degrees(np.arccos(dot_prod)) * 3600.0
        
        residuals.append(angle_arcsec)
        
        # Для выводим дни (считаем в секундах)
        dt_days = dt_sec / 86400.0
        
        status = "OK" if angle_arcsec < 2.0 else "POOR"
        if abs(dt_sec) < 0.1: status = "* ANCHOR *"
        
        print(f"{row_idx:<5} {t_obs:<15.1f} {dt_days:<15.4f} {angle_arcsec:<20.4f} {status}")

if residuals:
    rms = np.sqrt(np.mean(np.array(residuals)**2))
    print("-" * 85)
    print(f"RMS: {rms:.4f} arcsec")
