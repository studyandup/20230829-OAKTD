import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
import gym
import csv

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))



# def number_compute(number):
#     '''获取列表元素的总和,最大值,最小值,均值,中位数及方差'''
    
   
#     #对原始数据排序        可以对先对列表使用sort()方法排序，第一个及最后一个元素即是最大最小值。
#     number_list = number[:]#将原始数据复制一遍,操作数据时避免改动原始数据
#     number_list.sort()
    
#     #获取最大,最小值
#     max_n = number_list[len(number_list)-1]
#     min_n = number_list[0]
    
#     #获取总值        求列表元素的总和：使用for循环遍历列表，将其中的元素依次相加求和即可
#     sum_n = 0
#     for i in number_list:
#         sum_n += i
 
#     #获取均值        使用len()得出列表元素的个数，然后将得出的总和/元素个数即可
#     ave_n = sum_n/len(number_list)
    
#     #获取中位数
#     med_n = 0
#     t = int(len(number_list)/2)   #使用int()方法将结果转换为整数
#     if len(number_list)%2==0:    #判断元素的个数
#         med_n = (number_list[t-1]+number_list[t])/2    #根据元素个数找到中位数
#     else:
#         med_n = number_list[t]
    
#     #获取方差
#     var_n = number_list[0]
#     sum_MX = 0
#     for i in number_list:
#         sum_MX += (ave_n-i)*(ave_n-i)    #计算每个元素与均数的差的平方
#         var_n = sum_MX/len(number_list)    #更具公式计算出方差的平方
    
#     #储存结果        将结果封装进字典
#     dic_r = {"总值":sum_n,"最大值":max_n,"最小值":min_n,"均值":ave_n,"中位数":med_n,"方差":var_n}
#     print("计算结果:",dic_r)

def avg(path):
 

    # average=[]
    # with open(path, 'r') as f:
    #     reader = csv.reader(f)
    #     # print(type(reader))
    #     for row in reader:
    #         # print(row)
    #         average.append(row)
    # print(average)
    # # number_compute(average)

    # Sum=0
    

    run_reward = pd.read_csv(path)
    # run_reward = np.reshape(average,-1)
    # print(len(run_reward))
    a = run_reward .to_numpy()
    # print(a)
    mean = np.mean(a)
    std = np.var(a)
    median = np.median(a)
    # mean = np.mean(mean, axis=0)
    
    
    # return 0,0

    
    return mean, std , median
    
    
if __name__ == '__main__':
    # env = gym.make("Mountaincar-v0")
    # env.render()
    path1 = '../results/OAKTD_testingsteps.csv'
    path2 = '../results/TileCoding_testingsteps.csv'
    a , var_a , median_a= avg(path1)
    b , var_b , median_b= avg(path2)
    print(a)
    print(var_a)
    print(median_a)
    print(b)
    print(var_b)
    print(median_b)




