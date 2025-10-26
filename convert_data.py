import csv
import math

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

with (open("1I.txt", mode='r', encoding="UTF-8") as file):
    with open('data.csv', 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=';')
        datawriter.writerow(['time', 'x', 'y', 'z', 'code observatory'])
        start_year = 2017
        start_month = 10
        day = 14.43936
        start_total_sec = sum(i * 24 * 3600 for i in days[:start_month]) + day * 24 * 3600
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

            year = int(''.join(i for i in data[1] if i.isdigit()))
            month = int(data[2])
            day = float(data[3])
            total_sec = (year - start_year) * (365 * 24 * 3600) + \
                        sum(i * 24 * 3600 for i in days[:month]) + day * 24 * 3600
            sec = total_sec - start_total_sec

            if type_data != 's': # данные не со спутника Хабл
                ra = data[4:7]
                alpha = hms_to_deg(float(ra[0]), float(ra[1]), float(ra[2]))

                dec = data[7:10]
                sign_dec = (1, -1)['-' in dec[0]]
                delta = dms_to_deg(abs(float(dec[0])), float(dec[1]), float(dec[2]), sign_dec)

                x, y, z = calculate_rho_hat(alpha, delta)

                loc = data[-1][-3:]

                datawriter.writerow([f'{sec}', f'{x}', f'{y}', f'{z}',  f'{loc}'])