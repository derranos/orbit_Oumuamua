import numpy as np

def state_to_keplerian(r_vec, v_vec, mu):
    """
    Преобразует векторы положения и скорости в Кеплеровы элементы орбиты.
    
    Параметры:
    r_vec : array_like (x, y, z) в км
    v_vec : array_like (x, y, z) в км/с
    mu    : гравитационный параметр центрального тела (км^3/с^2)
    
    Возвращает словарь с элементами:
    a (большая полуось), e (эксцентриситет), i (наклонение),
    omega (аргумент перицентра), RAAN (долгота восходящего узла), nu (истинная аномалия)
    """
    r_vec = np.array(r_vec)
    v_vec = np.array(v_vec)
    
    # 1. Расстояние (r) и скорость (v)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # 2. Вектор удельного момента импульса (h = r x v)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # 3. Вектор линии узлов (n = k x h), где k = [0, 0, 1]
    k_vec = np.array([0, 0, 1])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    
    # 4. Вектор эксцентриситета (e_vec)
    # e = (1/mu) * ((v^2 - mu/r)*r - (r . v)*v)
    e_vec = (1/mu) * ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    e = np.linalg.norm(e_vec)
    
    # 5. Удельная механическая энергия (E) и Большая полуось (a)
    specific_energy = (v**2 / 2) - (mu / r)
    
    if abs(e - 1.0) > 1e-9:
        # Эллипс или гипербола
        a = -mu / (2 * specific_energy)
    else:
        # Парабола (a не определена, используют p)
        a = np.inf 

    # 6. Наклонение (i)
    # cos(i) = h_z / h
    i_rad = np.arccos(h_vec[2] / h)
    
    # 7. Долгота восходящего узла (Omega / RAAN)
    # cos(Omega) = n_x / n
    if n != 0:
        raan_rad = np.arccos(n_vec[0] / n)
        if n_vec[1] < 0:
            raan_rad = 2 * np.pi - raan_rad
    else:
        raan_rad = 0 # Для экваториальных орбит

    # 8. Аргумент перицентра (omega)
    # cos(omega) = (n . e) / (n * e)
    if n != 0 and e > 1e-9:
        omega_rad = np.arccos(np.dot(n_vec, e_vec) / (n * e))
        if e_vec[2] < 0:
            omega_rad = 2 * np.pi - omega_rad
    else:
        omega_rad = 0

    # 9. Истинная аномалия (nu)
    # cos(nu) = (e . r) / (e * r)
    if e > 1e-9:
        val = np.dot(e_vec, r_vec) / (e * r)
        val = np.clip(val, -1.0, 1.0) # Защита от ошибок округления
        nu_rad = np.arccos(val)
        if np.dot(r_vec, v_vec) < 0:
            nu_rad = 2 * np.pi - nu_rad
    else:
        nu_rad = 0

    # Конвертация в градусы
    res = {
        "a_km": a,
        "e": e,
        "i_deg": np.degrees(i_rad),
        "RAAN_deg": np.degrees(raan_rad),
        "omega_deg": np.degrees(omega_rad),
        "nu_deg": np.degrees(nu_rad),
        "period_days": 2 * np.pi * np.sqrt(a**3 / mu) / 86400 if a > 0 else None
    }
    return res

# --- Входные данные ---
# Гравитационный параметр Солнца (km^3/s^2)
# Если это не Солнце, замените на 398600.4418 для Земли
MU_SUN = 1.32712440018e11

with open("data1.txt", "r") as f:
        lines = f.readlines()
        t_epoch = float(lines[0].strip())
        
        # Читаем строки и превращаем в numpy массивы
        r_vec_epoch = np.array([float(x) for x in lines[1].split()])
        v_vec_epoch = np.array([float(x) for x in lines[2].split()])

r_input = r_vec_epoch
v_input = v_vec_epoch

# Вычисление
elements = state_to_keplerian(r_input, v_input, MU_SUN)

# --- Вывод результатов ---
print("-" * 30)
print(f"Большая полуось (a):  {elements['a_km']:.2f} км ({elements['a_km']/1.496e8:.3f} AU)")
print(f"Эксцентриситет (e):   {elements['e']:.6f}")
print(f"Наклонение (i):       {elements['i_deg']:.4f}°")
print(f"Долгота восх. узла (Ω): {elements['RAAN_deg']:.4f}°")
print(f"Аргумент перицентра (ω): {elements['omega_deg']:.4f}°")
print(f"Истинная аномалия (ν):  {elements['nu_deg']:.4f}°")
print("-" * 30)
if elements['period_days']:
    print(f"Орбитальный период:   {elements['period_days']:.2f} суток")