import math
import numpy as np

import thermalpower_

H =4
length_board_def = 13.4853
length_board = 6
def change_vector(v1,v2,angle1 = 0.00435/2):
    v = np.array(v1)

    # 方向向量1，这里假设为单位向量
    direction1 = np.array(v2)
    direction1 = direction1 / np.linalg.norm(direction1)  # 单位化

    # 构建旋转矩阵
    # 使用罗德里格斯公式构建旋转矩阵，绕 direction1 旋转 angle1 弧度
    def rotation_matrix(axis, theta):
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2)
        b, c, d = -axis * np.sin(theta / 2)
        return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    # 旋转向量 v
    rotated_v = np.dot(rotation_matrix(direction1, angle1), v)

    # print("原始向量 v:", v)
    # print("旋转后的向量:", rotated_v)
    return rotated_v
# if __name__ == '__main__':
#     change_vector()
def two_vector_get_ra(v1,v2):
    a = np.array(v1)
    b = np.array(v2)

    # 计算向量 a 和向量 b 的点积
    dot_product = np.dot(a, b)

    # 计算向量 a 和向量 b 的模长
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 计算夹角（弧度）
    cos_theta = dot_product / (norm_a * norm_b)
    theta_radians = np.arccos(cos_theta)
    return theta_radians
def line_circle_intersection2(A,B,C):
    # 圆的参数,圆心(0,0)
    r = 3.5  # 圆的半径
    # 计算二次方程的系数
    a = A ** 2 + B ** 2
    b = 2 * (A * C)
    c = C**2-B**2*r**2
    # 计算二次方程的判别式
    discriminant = b ** 2 - 4 * a * c
    # 检查是否有实数根
    if discriminant < 0:
        # print("直线和圆没有交点")
        return []
    elif discriminant == 0:
        x = -b / (2 * a)
        return [x]
    else:
        # 有两个交点
        x1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x2 = (-b - math.sqrt(discriminant)) / (2 * a)
        return [x1,x2]

def plane_equation_3d_to_2d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    # 计算方向向量
    direction_vector = (x2 - x1, y2 - y1, z2 - z1)
    # 计算法向量
    a, b, c = direction_vector
    a0,b0,c0 = 0,0,0
    if a!= 0:
        t = -x1/a
        y_target = y1 + t*b
        c0 = - y_target
        b0 = 1
        a0 = b/a
    else:
        a0 = 1
        c0 = -x1
    return a0,b0,c0,direction_vector

def judge_z(point1,x,direction_vector):#已知x,y求z
    x1, y1, z1 = point1
    a, b, c = direction_vector
    if a != 0:
        z = (x - x1 ) *(c/a) + z1
    else:
        a = 1e-8
        z = (x - x1) * (c / a) + z1
    return z
def two_point_get_intersection_point(x,y):#通过两个点获取是否相交，是集成函数
    point1 = (x,y,H)
    point2 = (0,0,80)
    a,b,c,direction_vector = plane_equation_3d_to_2d(point1,point2 )
    # print(a,b,c)
    solutions = line_circle_intersection2(a, b, c)
    # print(len(solutions))
    if len(solutions) >=1:
        print("坐标1为",solutions[0][0], solutions[0][1])
        print("高度为",judge_z(point1, solutions[0][0], direction_vector))
    if len(solutions) >=2:
        print("坐标2为",solutions[1][0], solutions[1][1])
        print("高度为",judge_z(point1, solutions[1][0], direction_vector))
def one_point_and_vector_intersection_point(x,y,direction_vector):#通过一个点和方向向量判断是否相交，集成函数
    point1 = (x,y,H)
    point2 = (x+direction_vector[0],y+direction_vector[1],H+direction_vector[2])
    a, b, c, direction_vector = plane_equation_3d_to_2d(point1, point2)
    # print(a, b, c)

    solutions = line_circle_intersection2(a, b, c)

    num = 0
    if len(solutions) >= 1:
        height = judge_z(point1, solutions[0], direction_vector)
        if height > 76 and height < 84:
            num += 1
    if len(solutions) >= 2:
        height = judge_z(point1, solutions[1], direction_vector)
        if height > 76 and height < 84:
            num += 1




    if len(solutions) >= 1:
        if num>0:
            return True  # 有交点
    return False#无交点
def join_two_vector(vector1,vector2):
    vector1_mod = math.sqrt(vector1[0]**2+vector1[1]**2+vector1[2]**2)
    vector2_mod = math.sqrt(vector2[0]**2+vector2[1]**2+vector2[2]**2)
    a1,b1,c1 = vector1[0]/vector1_mod,vector1[1]/vector1_mod,vector1[2]/vector1_mod
    vector1 = (a1,b1,c1)
    a2,b2,c2 = vector2[0]/vector2_mod,vector2[1]/vector2_mod,vector2[2]/vector2_mod
    vector2 = (a2,b2,c2)
    vector_fa = ((vector1[0]+vector2[0])/2,(vector1[1]+vector2[1])/2,(vector1[2]+vector2[2])/2)
    # print(vector_fa)
    return vector_fa
def get_v_to_z(vector):
    vector_v_to_z = (-vector[1],vector[0],0)
    return vector_v_to_z
def get_v_vector(vector):
    vector_v = (vector[0],vector[1],-(vector[0]**2+vector[1]**2)/vector[2])
    return vector_v
def vector_normalization(vector1):
    vector1_mod = math.sqrt(vector1[0]**2+vector1[1]**2+vector1[2]**2)
    a1,b1,c1 = vector1[0]/vector1_mod,vector1[1]/vector1_mod,vector1[2]/vector1_mod
    vector1 = (a1,b1,c1)
    return vector1


def find_intersection(m1, m2, x1, y1, x2, y2):
    # 计算交点的 x 坐标
    x = (y2 - y1 + m1 * x1 - m2 * x2) / (m1 - m2)

    # 使用 x 坐标代入直线1的方程，或者直线2的方程来计算 y 坐标
    y = y1 + m1 * (x - x1)
    # print(x,y)
    return x, y


def judge_and_return_shadow(x,y,vector_solar):
    a,b,c= vector_solar[0],vector_solar[1],vector_solar[2]
    if a!= 0:
        k1 = b/a
    else:
        k1 =999999
    if k1==0:
        k1 = 1e-8
    k2 = -1/k1
    theta = math.asin(b/math.sqrt(b**2+a**2))
    point1 = (3.5*math.cos(theta+math.pi/2),3.5*math.sin(theta+math.pi/2))
    point2 = (3.5*math.cos(theta+(math.pi/2)*3),3.5*math.sin(theta+(math.pi/2)*3))
    x1, y1 = find_intersection(k1,k2,point1[0],point1[1],x,y)
    x2, y2 = find_intersection(k1,k2,point2[0],point2[1],x,y)
    if (x1-x)*(x2-x)<0:
        if math.sqrt(x**2+y**2)<(84*abs(math.sqrt(a**2+b**2)/c)):
            return 0
    return 1
def cal_shadow_2(vector_solar,vector_v):
    if vector_v[2]<0:
        a,b,c = -vector_v[0],-vector_v[1],-vector_v[2]
    else:
        a,b,c = vector_v[0],vector_v[1],vector_v[2]
    vector_v_2 = (a,b,c)
    # print(vector_v_2)
    vector_v_ground = (vector_v_2[0],vector_v_2[1],0)
    vector_solar_ground = (vector_solar[0],vector_solar[1],0)
    theta_solar = two_vector_get_ra(vector_solar,vector_solar_ground)
    theta_v = two_vector_get_ra(vector_v_2,vector_v_ground)
    theta_solar_top = math.pi - theta_solar -theta_v
    L2 = length_board_def *(math.sin(theta_solar)/math.sin(theta_solar_top))
    if L2>length_board:
        return 1
    return L2/length_board
# if __name__ == '__main__':
#     cal_shadow_2((0,2,1),(0,2,-1))
def cal_ntrunc(x,y,sin_solar_elevation,cos_solar_pos_ang):
    point1 = (x, y, H)
    point2 = (0, 0, 80)
    a0, b0, c0, direction_vector = plane_equation_3d_to_2d(point1, point2)
    vector_solar = thermalpower_.solar_dir(sin_solar_elevation,cos_solar_pos_ang)
    vector_normal = join_two_vector(direction_vector,vector_solar)
    vector_v_to_z = get_v_to_z(vector_normal)
    vector_v_to_z = vector_normalization(vector_v_to_z)
    vector_v = get_v_vector(vector_normal)
    vector_v =vector_normalization(vector_v)
    np_point = np.array(point1)
    np_vector_v = np.array(vector_v)
    np_vector_v_to_z = np.array(vector_v_to_z)
    total_num = 36*4
    right_num = 0

    for i in range(-3, 3):
        np_point_1 = np_point + (i+0.5)* np_vector_v

        for j in range(-3, 3):
            np_point_2 = np_point_1 + (j+0.5) * np_vector_v_to_z
            a10,b10,c10 =direction_vector[0],direction_vector[1],direction_vector[2]
            # if one_point_and_vector_intersection_point(np_point_2[0], np_point_2[1], direction_vector):
            #     # a+=1
            #     right_num += 1
            direction_vector_new = change_vector(direction_vector,(b10,-a10,0),(0.00435 *1)/ 3)
            if one_point_and_vector_intersection_point(np_point_2[0], np_point_2[1], direction_vector_new):
                right_num += 1
                # b+=1
            direction_vector_new = change_vector(direction_vector,(b10,-a10,0),(-0.00435 *1)/ 3)
            if one_point_and_vector_intersection_point(np_point_2[0], np_point_2[1], direction_vector_new):
                right_num += 1
                # b+=1

            direction_vector_new = change_vector(direction_vector, (a10,b10,-(a10**2+b10**2)/c10),(0.00435 *1)/ 3)
            if one_point_and_vector_intersection_point(np_point_2[0], np_point_2[1], direction_vector_new):
                right_num += 1
                # b+=1
            direction_vector_new = change_vector(direction_vector, (a10,b10,-(a10**2+b10**2)/c10), (-0.00435 *1)/ 3)
            if one_point_and_vector_intersection_point(np_point_2[0], np_point_2[1], direction_vector_new):
                right_num += 1
                # b+=1
            # if a!=0 and (b== 4):
            #     print("成功")
            # else:
            #     print("不")
    v1 = np.array(vector_solar)
    v2 = np.array(vector_normal)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算夹角的余弦值
    cosine_theta = dot_product / (norm_v1 * norm_v2)
    # 计算夹角（以弧度为单位）
    theta_radians = np.arccos(cosine_theta)
    n_sb_1 = judge_and_return_shadow(x,y,vector_solar)
    n_sb_2 = cal_shadow_2(vector_solar,vector_v)
    return right_num/total_num,math.cos(theta_radians),n_sb_1*n_sb_2
if __name__ == '__main__':
    # print(1745*36)
    print((546.3900156830103*1745*36)/1e6)
    # line_circle_intersection2(0,1,-3)
# 4.15~4.35