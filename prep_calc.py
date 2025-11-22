import csv
import numpy as np
from astropy import units as u
from astropy import constants as const
from solver import solve_3x3_cramer
import pickle



# Гравитационный параметр Солнца (k^2) в км^3/с^2
GM_SUN_KM3_S2 = const.GM_sun.to_value('km3/s2')

# Скорость света в км/с
C_KM_S = const.c.to_value('km/s')


CSV_FILENAME = 'data.csv'

print(f"Чтение файла {CSV_FILENAME}...")

all_t_sec = []  # Список для t_i (сек. от t0)
all_rho_hats = []  # Список для rho_hat_i (направление)
all_R_vecs = []  # Список для R_i (вектор наблюдателя от Солнца)


with open(CSV_FILENAME, 'r', encoding='utf-8') as f:
    # (заголовок: Date;time_in_sec;x;y;z;code observatory;R_x;R_y;R_z)
    header = f.readline()

    reader = csv.reader(f, delimiter=';')

    for row in reader:
        if not row or len(row) < 9: continue  # Пропускаем пустые/неполные строки

        # Читаем данные из колонок
        t_sec = float(row[1])

        rho_hat = np.array([
            float(row[2]),  # x
            float(row[3]),  # y
            float(row[4])  # z
        ])

        R_vec = np.array([
            float(row[6]),  # R_x
            float(row[7]),  # R_y
            float(row[8])  # R_z
        ])

        all_t_sec.append(t_sec)
        all_rho_hats.append(rho_hat)
        all_R_vecs.append(R_vec)


# возьмем первое, среднее и последнее наблюдения
idx1 = 10
idx2 = 20  # Целочисленное деление
idx3 = 35  # Последний элемент

# Извлекаем данные для 3-х точек
t1, t2, t3 = all_t_sec[idx1], all_t_sec[idx2], all_t_sec[idx3]
R1, R2, R3 = all_R_vecs[idx1], all_R_vecs[idx2], all_R_vecs[idx3]
rho_hat1, rho_hat2, rho_hat3 = all_rho_hats[idx1], all_rho_hats[idx2], all_rho_hats[idx3]

# Вычисляем разницы времени (в секундах)
tau1_obs = t1 - t2
tau3_obs = t3 - t2
tau_total_obs = t3 - t1

# Аппроксимируем c_i как соотношения времени
c1 = tau3_obs / tau_total_obs
c3 = -tau1_obs / tau_total_obs



print(f"Запуск итераций Метода Гаусса")

# Хранит магнитуду (длину) вектора r2 из предыдущей итерации.
# Используется для проверки, насколько изменилось r2, чтобы определить сходимость
r2_mag_old = 0.0

# Максимальное количество итераций, которое выполнит цикл,
# прежде чем остановиться (даже если решение не сошлось). Предотвращает бесконечный цикл.
max_iterations = 1000

# Порог сходимости (в километрах). Если изменение магнитуды r2
# между итерациями меньше этого значения, считаем, что решение найдено.
tolerance = 1e-15  # (in km)

# Коэффициенты Лагранжа (f и g). Они связывают положения и скорости
# в разные моменты времени. Будут пересчитываться на каждой итерации.
f1, g1, f3, g3 = 0, 0, 0, 0

# Знаменатель, используемый при вычислении c1, c3 и финальной скорости v2.
# Будет пересчитываться на каждой итерации.
D = 0

# Гелиоцентрические векторы положения (X, Y, Z в км) для трех наблюдений.
# Вычисляются внутри цикла на каждой итерации.
r1_vec, r2_vec, r3_vec = None, None, None

# Гелиоцентрический вектор скорости (Vx, Vy, Vz в км/с) для среднего наблюдения (t2).
# Вычисляется только ОДИН РАЗ после успешного завершения цикла. Начинаем с None.
v2_vec = None

# Флаг (True/False), показывающий, успешно ли сошелся итерационный процесс
converged = False

# Хранит абсолютную разницу между r2_mag текущей и r2_mag_old предыдущей итерации
error = float('inf')  # Initialize error

for i in range(max_iterations):

    # c1 * p1 * p`1 - p2 * p`2 + c3 * p3 * p`3 = R2 - c1*R1 - c3*R3

    #(vector b)
    b = R2 - (c1 * R1) - (c3 * R3)

    A = np.column_stack([
        c1 * rho_hat1,
        -rho_hat2,
        c3 * rho_hat3
    ])

    # A * [rho1, rho2, rho3] = b
    rho_scalars = solve_3x3_cramer(A, b)

    rho1, rho2, rho3 = rho_scalars[0], rho_scalars[1], rho_scalars[2]

    # t_dyn = t_obs - (rho / c)
    t1_dyn_sec = t1 - (rho1 / C_KM_S)
    t2_dyn_sec = t2 - (rho2 / C_KM_S)
    t3_dyn_sec = t3 - (rho3 / C_KM_S)

    tau1_sec = t1_dyn_sec - t2_dyn_sec
    tau3_sec = t3_dyn_sec - t2_dyn_sec

    # r_i = rho_i * rho_hat_i + R_i
    r1_vec = rho1 * rho_hat1 + R1
    r2_vec = rho2 * rho_hat2 + R2
    r3_vec = rho3 * rho_hat3 + R3

    r2_mag = np.linalg.norm(r2_vec)

    error = abs(r2_mag - r2_mag_old)
    print(f"  Iteration {i + 1}: r2 = {r2_mag:.1f} km, (change = {error:.2e} km)")

    if error < tolerance:
        print(f"Решение сошлось после {i + 1} итераций .\n")
        converged = True
        break  # SUCCESS

    if i == max_iterations - 1:
        print("Ошибка. Не сошлось")
        break  # FAIL

    r2_mag_old = r2_mag

    # u = mu / r^3
    u_param = GM_SUN_KM3_S2 / (r2_mag ** 3)

    f1 = 1.0 - 0.5 * u_param * (tau1_sec ** 2)
    g1 = tau1_sec - (1.0 / 6.0) * u_param * (tau1_sec ** 3)

    f3 = 1.0 - 0.5 * u_param * (tau3_sec ** 2)
    g3 = tau3_sec - (1.0 / 6.0) * u_param * (tau3_sec ** 3)

    D = f1 * g3 - f3 * g1

    if abs(D) < 1e-15:
        print("Error")
        break

    c1 = g3 / D
    c3 = -g1 / D


if converged and r1_vec is not None and r3_vec is not None and abs(D) > 1e-15:

    # v2 = (f1*r3 - f3*r1) / D
    v2_vec = (f1 * r3_vec - f3 * r1_vec) / D

    print(f"Position r2 (X,Y,Z): {r2_vec} km")
    print(f"Velocity v2 (X,Y,Z): {v2_vec} km/s")

    r_final = r2_vec * u.km
    v_final = v2_vec * u.km / u.s

    data = {'r_vec': list(r_final), 'v_final': list(v_final)}
    dbfile = open('data.txt', 'w+')
    dbfile.write(str(t2) + '\n')

    dbfile.write(' '.join([str(i) for i in r2_vec]) + '\n')
    dbfile.write(' '.join([str(i) for i in v2_vec]) + '\n')              
    dbfile.close()