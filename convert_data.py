import csv
import math
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import EarthLocation, get_body_barycentric, GCRS
from astropy.time import Time

R_EARTH_METERS = const.R_earth.to_value(u.m)

days_in_month = {
    1: 31,  # Январь
    2: 28,  # Февраль
    3: 31,  # Март
    4: 30,  # Апрель
    5: 31,  # Май
    6: 30,  # Июнь
    7: 31,  # Июль
    8: 31,  # Август
    9: 30,  # Сентябрь
    10: 31, # Октябрь
    11: 30, # Ноябрь
    12: 31  # Декабрь
}

days = list(days_in_month.values())

def hms_to_deg(h, m, s):
    """
    Конвертирует Прямое восхождение (Часы:Минуты:Секунды)
    в десятичные градусы.
    Формула: α = (Ч+М/60+С/3600)*15
    """
    return (h + m/60.0 + s/3600.0) * 15.0

def dms_to_deg(d, m, s, sign=1.0):
    """
    Конвертирует Склонение (Градусы:Минуты:Секунды)
    в десятичные градусы.
    Формула: δ = ±(Г+М/60+С/3600)
    """
    return sign * (abs(d) + m/60.0 + s/3600.0)

def calculate_rho_hat(alpha_deg, delta_deg):
    """
    Преобразует (α, δ) в векторы единичного направления.
    Формулы:
    x = cos(δ) * cos(α)
    y = cos(δ) * sin(α)
    z = sin(δ)
    """
    # Тригонометрические функции в Python (math) ожидают углы в радианах
    alpha_rad = math.radians(alpha_deg)
    delta_rad = math.radians(delta_deg)

    x = math.cos(delta_rad) * math.cos(alpha_rad)
    y = math.cos(delta_rad) * math.sin(alpha_rad)
    z = math.sin(delta_rad)

    return x, y, z



def get_ground_observer_heliocentric_vector(obs_time, obs_code_log):
    """
    Вычисляет гелиоцентрический вектор (X,Y,Z) в км для наблюдателя,

    Шаг 1: Конвертирует (Long, cos, sin) -> (X,Y,Z)_ITRF (в метрах)
    Шаг 2: Создает 'EarthLocation' из этих (X,Y,Z)_ITRF
    Шаг 3: Конвертирует 'EarthLocation' -> (X,Y,Z)_GCRS (вектор отн. Земли в АЕ)
    Шаг 4: Вычисляет (X,Y,Z) Земли отн. Солнца (в АЕ)
    Шаг 5: Складывает (Шаг 3 + Шаг 4) и конвертирует в км.
    """

    lon_deg, cos_val, sin_val = None, None, None

    with open("code_observatory.txt", mode='r') as file_obs:
        for line_obs in file_obs.readlines():
            line_obs = line_obs.split()
            code = line_obs[0]
            if code == obs_code_log:
                lon_deg, cos_val, sin_val = float(line_obs[1]), float(line_obs[2]), float(line_obs[3])

    # долготу в радианы
    lon_rad = math.radians(lon_deg)

    # cos_val - это (rho * cos(phi'))
    # sin_val - это (rho * sin(phi'))

    # X_itrf = R_Земли * (rho*cos(phi')) * cos(Долготы)
    X_itrf_meters = R_EARTH_METERS * cos_val * math.cos(lon_rad)

    # Y_itrf = R_Земли * (rho*cos(phi')) * sin(Долготы)
    Y_itrf_meters = R_EARTH_METERS * cos_val * math.sin(lon_rad)

    # Z_itrf = R_ЗEMЛИ * (rho*sin(phi'))
    Z_itrf_meters = R_EARTH_METERS * sin_val


    # используем .from_geocentric() для создания из (X,Y,Z)
    try:
        obs_location_obj = EarthLocation.from_geocentric(
            x=X_itrf_meters * u.m,
            y=Y_itrf_meters * u.m,
            z=Z_itrf_meters * u.m
        )
    except Exception as e:
        print(f"Ошибка создания EarthLocation для {obs_code_log}: {e}")
        raise

    # Конвертируем в GCRS (небесная система, отн. Земли)
    # .get_gcrs(obs_time) учитывает вращение Земли
    # .cartesian.xyz возвращает (x,y,z) в АЕ
    observatory_geocentric_vec_au = obs_location_obj.get_gcrs(obs_time).cartesian.xyz

    # Получаем гелиоцентрический вектор Земли
    # Вектор(Солнце->Земля) в АЕ
    earth_pos_bary = get_body_barycentric('earth', obs_time)
    sun_pos_bary = get_body_barycentric('sun', obs_time)
    earth_heliocentric_vec_au = (earth_pos_bary - sun_pos_bary).xyz

    # Складываем и конвертируем в км
    # Вектор(Солнце->Земля) + Вектор(Земля->Обсерватория)
    final_vec_au = earth_heliocentric_vec_au + observatory_geocentric_vec_au
    final_vec_km = final_vec_au.to_value(u.km)

    return final_vec_km



with (open("1I.txt", mode='r', encoding="UTF-8") as file):
    with open('data.csv', 'w', newline='') as csvfile:
        #---------------------------
        datawriter = csv.writer(csvfile, delimiter=';')
        datawriter.writerow(['Date', 'time_in_sec', 'x', 'y', 'z', 'code observatory', 'R_x', 'R_y', 'R_z'])
        #---------------------------
        start_year = 2017
        start_month = 10
        day = 14.43936
        start_total_sec = sum(i * 24 * 3600 for i in days[:start_month - 1]) + day * 24 * 3600
        #---------------------------
        # для дальнейшего вычисления координат гелиоцентрического положения обсерватории

        t_base = Time(f"{start_year}-{start_month}-{int(day)} 00:00:00", scale='utc')
        t0 = t_base + ((day - int(day)) * u.day)

        conter = 0
        while line := file.readline():

            index_plus = line.find('+')
            if line[index_plus - 1] != ' ' and index_plus != -1:
                line = line[:index_plus] + ' ' + line[index_plus:] # если плюс и пробел слиплись в данных
                data = line.split()
                day, coord = data[3][:-2], data[3][-2:] # если день и координата слиплись в данных
                data = data[:3] + [day, coord] + data[4:]
            else:
                data = line.split()

            type_data = data[1][0]

        
            # year = int(''.join(i for i in data[1] if i.isdigit())) # избавляюсь от букв в годе
            year = 2017
            month = int(data[2])
            day = float(data[3])
            total_sec = (year - start_year) * (365 * 24 * 3600) + \
                        sum(i * 24 * 3600 for i in days[:month - 1]) + day * 24 * 3600
            sec = total_sec - start_total_sec

            if type_data.lower() != 's': # данные не со спутника Хабл
                ra = data[4:7]
                alpha = hms_to_deg(float(ra[0]), float(ra[1]), float(ra[2]))

                dec = data[7:10]
                sign_dec = (1, -1)['-' in dec[0]]
                delta = dms_to_deg(abs(float(dec[0])), float(dec[1]), float(dec[2]), sign_dec)

                # единичный вектор
                x, y, z = calculate_rho_hat(alpha, delta)

                # код обсерватории
                obs_code = data[-1][-3:]

                # вектор наблюдателя R_i

                t_i = t0 + (sec * u.second)

                R_i = get_ground_observer_heliocentric_vector(t_i, obs_code)


                datawriter.writerow([f"{t_i}",f'{sec}', f'{x}', f'{y}', f'{z}',  f'{obs_code}', f'{R_i[0]}', f'{R_i[1]}', f'{R_i[2]}'])