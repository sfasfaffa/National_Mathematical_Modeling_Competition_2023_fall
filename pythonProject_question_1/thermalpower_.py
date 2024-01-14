import math
import openpyxl
import math_cal
# 打开 Excel 文件
# 关闭 Excel 文件
import math
import datetime
def cal_Dni(sin_s_elevation):
    H=3
    G0 = 1366 # w/m**2
    a = 0.4237 - 0.00821 * (6 - H) ** 2
    b = 0.5055 + 0.00595 * (6.5 - H) ** 2
    c = 0.2711 + 0.01858 * (2.5 - H) ** 2
    DNI = G0*(a+b*math.exp(-c/sin_s_elevation))
    return DNI
def cal_sin_solar_elevation(solar_declination,latitude,solar_hour_angle):
    sin_alpha_s = math.sin(solar_declination) * math.sin(latitude) + math.cos(solar_declination) * math.cos(
        latitude) * math.cos(solar_hour_angle)
    return sin_alpha_s

def calculate_solar_declination(i):
    # 计算太阳赤纬角的sin值
    date = datetime.date(2023, i, 21)
    # 计算以春分作为第0天的天数
    spring_equinox = datetime.date(date.year, 3, 21)  # 假设春分是3月21日
    day_of_year = (date - spring_equinox).days
    # 计算太阳赤纬角，以度数表示
    delta_degrees_sin =math.sin(2*math.pi*day_of_year/365)*math.sin(2*math.pi*23.45/360)
    delta_degrees = math.asin(delta_degrees_sin)
    # 23.45 * math.sin(math.radians(360 * (day_of_year - 81) / 365))
    return delta_degrees

def solar_hour_angle(hour, minute):
    # 计算太阳时角，以弧度表示
    total_minutes = hour * 60 + minute
    solar_hour_angle_rad = (math.pi / 12) * (total_minutes / 60 - 12)
    return solar_hour_angle_rad
#太阳方位角
def cos_solar_position_angle(solar_delta_angle,latitude,sin_solar_elevation):
    cos_solar_pos_angle = (math.sin(solar_delta_angle) - sin_solar_elevation * math.sin(latitude))/(math.cos(math.asin(sin_solar_elevation))*math.cos(latitude))
    if cos_solar_pos_angle<-1:
        cos_solar_pos_angle = -1
    if cos_solar_pos_angle>1:
        cos_solar_pos_angle = 1
    return cos_solar_pos_angle

def judgeTime(i):
    hour = 0
    minute = 0
    if i == 1:
        hour = 9
        minute = 0
    elif i == 2:
        hour = 10
        minute = 30
    elif i == 3:
        hour = 12
        minute=0
    elif i ==4:
        hour = 13
        minute = 30
    elif i == 5:
        hour = 15
        minute = 0
    return hour,minute
def solar_dir(sin_solar_elevation,cos_solar_pos_ang):
    z = sin_solar_elevation
    x_y = math.cos(math.asin(sin_solar_elevation))
    x = math.sin(math.acos(cos_solar_pos_ang))
    y = cos_solar_pos_ang*x_y
    return (x,y,z)
def cal_nat(x,y):
    distance = math.sqrt(x**2+y**2+76**2)
    n_at = 0.99321 - 0.0001176*distance + 1.97*1e-8*(distance**2)
    return n_at
if __name__ == '__main__':
    latitude_de = 39.4
    workbook = openpyxl.load_workbook('附件.xlsx')
    # 选择 Sheet1
    sheet = workbook['Sheet1']
    n_trunc_year_a = 0
    n_at_year_a = 0
    n_sb_year_a = 0
    n_cos_year_a = 0
    hot_power_year_a = 0
    n_i_year_a =0
    for i in range(1, 13):#遍历12个月
        # print(calculate_solar_declination(i))
        n_trunc_A, n_cos_A, n_sb_A, n_at_A ,n_i_A= 0, 0, 0, 0,0
        themal_power_per_m = 0
        DNI = 0
        for j in range(1, 6):#遍历五个时间
            hour, minute = judgeTime(j)
            # print(hour, minute)
            s_hour_angle = solar_hour_angle(hour, minute)  # 计算太阳时角，以弧度表示
            solar_delta_angle = calculate_solar_declination(i)  # 太阳赤纬角
            sin_solar_elevation = cal_sin_solar_elevation(solar_delta_angle, math.radians(latitude_de),
                                                          s_hour_angle)  # 太阳高度角
            cos_solar_pos_ang = cos_solar_position_angle(solar_delta_angle, math.radians(latitude_de),
                                                         sin_solar_elevation)  # 太阳方位角
            DNI = cal_Dni(sin_solar_elevation)
            for row_index, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
                fir_column_value = row[0]
                se_column_value = row[1]  # 第2列的值
                # print(row_index, row[0], row[1])
                Ai = 6*6
                n_trunc , n_cos, n_sb= math_cal.cal_ntrunc(fir_column_value,se_column_value,sin_solar_elevation,cos_solar_pos_ang)
                n_at = cal_nat(fir_column_value,se_column_value)
                ni = n_trunc*n_cos*0.92*n_at*n_sb
                themal_power_per_m += DNI*Ai*ni
                n_trunc_A += n_trunc
                n_cos_A += n_cos
                n_at_A += n_at
                n_sb_A += n_sb
                n_i_A += ni
                # if(row_index%1700 == 0):
                #     print("完成第",i,"月,第",j,"时,第",row_index,"行")
        print("完成第",i,"月")
        print("DNI:",DNI)
        print("n_trunc_A:",n_trunc_A/(1745*5),"n_cos_A:", n_cos_A/(1745*5), "n_sb_A:",n_sb_A/(1745*5), "n_at_A:",n_at_A/(1745*5))
        print("ni_A:",n_i_A/(1745*5))
        n_trunc_year_a+=n_trunc_A/(1745*5)
        n_cos_year_a += n_cos_A/(1745*5)
        n_sb_year_a += n_sb_A/(1745*5)
        n_at_year_a += n_at_A/(1745*5)
        n_i_year_a += n_i_A/(1745*5)
        print("themal_power_per_m:",themal_power_per_m/(36*1745*5))
        hot_power_year_a += themal_power_per_m/(36*1745*5)
    print("年均：")
    print("n_trunc_A:",n_trunc_year_a/12,"n_cos_A:", n_cos_year_a/(12), "n_sb_A:",n_sb_year_a/(12), "n_at_A:",n_at_year_a/(12))
    print("n_i_year_a:",n_i_year_a/12)
    print("hot_power_year_a:",hot_power_year_a/12)

