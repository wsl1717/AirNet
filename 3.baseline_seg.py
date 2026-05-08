# -*- coding: utf-8 -*-
"""
Created on Oct 10,2025
@author: sl.wang
分段拟合专用框架
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import time
import torch
import numpy as np
import copy
from torch.autograd import Variable
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from model.ResUNet_2500 import ResUNet_2500
from scipy.stats import kurtosis
import matplotlib.pyplot as pl

pl.rcParams['font.sans-serif'] = ['SimHei'] 
pl.rcParams['axes.unicode_minus'] = False    

def normalization(data):
    data = data/np.max(data)
    return data

def distance_Reciprocal(a,b):
    d_all=a**2+b**2
    d=d_all**0.5
    d=1/d
    return d

def weight_result(data,c):
    m = (c-1)//2
    n=len(data)
    new_data= [0 for x in range(n)]
    for i in range(0,n):
        w_sum=0
        if i <= m - 1:
            lx = 0
            ly = i + m
            p_sum=0
            w_sum=w_sum+data[0]*(m-i)
            a=0
            happens = [0 for x in range(ly-lx+1)]
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range(lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
        elif i >= n - m:
            lx = i - m
            ly = n-1
            p_sum=0
            w_sum=w_sum+data[n-1]*(m-n+i)
            a=0
            happens = [0 for x in range(ly-lx+1)]
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range (lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
        else:
            lx = i - m
            ly = i + m
            p_sum=0
            happens = [0 for x in range(ly-lx+1)]
            a=0
            for j in range(lx,ly+1):
                w_sum=w_sum+data[j]
            avg=w_sum/c
            if avg in data[lx:ly+1]:
                new_data[i]=avg
            else:
                for k in range (lx,ly+1):
                    p_sum=p_sum+distance_Reciprocal(data[k],avg)
                for k in range (lx,ly+1):
                    happens[a]=data[k]*(distance_Reciprocal(data[k],avg)/p_sum)
                    new_data[i]=new_data[i]+happens[a]
                    a=a+1
    return new_data

def derivative_single(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    return result

def derivative(data):
    result = [data[i] - data[i - 1] for i in range(1, len(data))]
    result.insert(0, result[0])
    result2= [result[i] - result[i - 1] for i in range(1, len(result))]
    return result,result2

def find_extreme(data,d_data,d2_data):
    big = []
    small=[]
    big_mis=[]
    data_temp=copy.deepcopy(data)
    for i in range(len(d_data) - 1):
        if (d_data[i] < 0 and d_data[i + 1] > 0) :
            small.append(i)
    for i in range(len(small) - 1):
        for j in range(small[i], small[i + 1]):
            if (d_data[j] > 0 and d_data[j + 1] < 0):
                big.append([[small[i], small[i + 1]], j])
    #print(big)
    big_premis=[]
    big_premis_t=[]
    for s in range(len(big)):
        half_k=(big[s][0][1]-big[s][0][0])//2
        if half_k>=3:
            half_k=2
        for w in range(half_k):
            if big[s][1]+w<len(d2_data) and big[s][1]-w>0:
                if d2_data[big[s][1]+w]>=0 or d2_data[big[s][1]-w]>=0:
                    big_premis.append(s)
    big_premis=list(set(big_premis))
    for s in big_premis:
        big_premis_t.append(big[s])
    #print(big_premis_t)
    for s in  big_premis_t:
        big.remove(s)
    for m in range(len(big)-1):
        if data[big[m][0][1]]>data[big[m][0][0]]:
            if data[big[m][0][1]]-data[big[m][0][0]]>0:
                list1=[]
                for f in range(big[m][1],big[m][0][1]):
                    list1.append(abs(data[f]-data[big[m][0][1]]))
                big[m][0][1]=list1.index(min(list1))+big[m][1]
                del list1
        else:
                if data[big[m][0][0]]-data[big[m][0][1]]>0:
                    list1=[]
                    for f in range(big[m][0][0],big[m][1]):
                        list1.append(abs(data[f]-data[big[m][0][1]]))
                    big[m][0][0]=list1.index(min(list1))+big[m][0][0]
                    del list1
    #print('----------------')
    m=5
    for y in range(len(big)-1):
        q=big[y][0][1]-big[y][0][0]
        rush=q//2-int(q//m)
        data_temp1=data[big[y][0][0]+rush:big[y][1]+1]
        data_temp2=data[big[y][1]:big[y][0][1]+1-rush]
        flag1=0
        flag2=0
        for h in range(len(data_temp1)-1):
            if data_temp1[h+1]-data_temp1[h]<0:
                flag1=flag1+1
        for x in range(len(data_temp2)-1):
            if data_temp2[x+1]-data_temp2[x]>0:
                flag2=flag2+1
    big_new=[]
    small_new=[]
    for b in range(len(big)):
        small_new.append(big[b][0][0])
        small_new.append(big[b][0][1])
    for u in range(len(big)):
        big_new.append(big[u][1])
    for r in range(0,len(small_new)-1,2):
        q=small_new[r+1]-small_new[r]
        for t in range(1,int(q//m)+1):
            big_new.append(big_new[r//2]+t)
            big_new.append(big_new[r//2]-t)
    big_new.sort()
    big_new_searching=[]
    for i in big_new:
        if i > len(data)-2 or i <1:
            big_new_searching.append(i)
    for i in big_new_searching:
        big_new.remove(i)
    return big_new,data_temp

def PEER(data,c):
    d_data,d2_data=derivative(data)
    big_new,data_temp=find_extreme(data,d_data,d2_data)
    new_data=weight_result(data_temp,c)
    for l in big_new:
        new_data[l]=data[l]
    return new_data

def detect_delimiter(file_path):
    with open(file_path, 'r',encoding='utf-8-sig') as file:
        first_line = file.readline().strip()
        if ',' in first_line:
            return ','
        elif '\t' in first_line:
            return '\t'
        else:
            return ' '

def read(filename):
    file = open(filename, encoding='utf-8-sig')
    data_lines = file.readlines()
    file.close()
    orign_keys = []
    orign_values = []
    delimiter=detect_delimiter(filename)
    for data_line in data_lines:
        pair = data_line.split(delimiter)
        key = float(pair[0])
        value = float(pair[1])
        orign_keys.append(key)
        orign_values.append(value)
    return orign_keys, orign_values

def write(filename, files, values):
    file = open(filename, 'w')
    for k, v in zip(files, values):
        file.write(str(k) + " " + str(v) + "\n")
    file.close()

def WhittakerSmooth(x,w,lambda_,differences=2):
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] 
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def auto_select_lambda(spectrum_raw, weights, differences=2, 
                      lambda_range=(1e1, 1e11), num_candidates=20):
    
    min_lambda, max_lambda = lambda_range
    log_min = np.log10(min_lambda)
    log_max = np.log10(max_lambda)

    log_lambdas = np.arange(log_min, log_max + 1, 1.0) 
    lambda_candidates = [10**log_lambda for log_lambda in log_lambdas]
    
    best_lambda = lambda_candidates[0]
    best_score = float('inf')
    lambda_scores = []
    
    for current_lambda in lambda_candidates:
        baseline_temp = WhittakerSmooth(spectrum_raw, weights, current_lambda, differences)
        corrected_temp = spectrum_raw - baseline_temp

        # pl.plot(spectrum_raw, '-b', label='PEER去噪')
        # pl.plot(baseline_temp, '-r', label='基线')  
        # pl.title(f'λ={current_lambda:.0e}')
        # pl.legend()
        # pl.show()
        
        neg_points = corrected_temp[corrected_temp < 0]
        neg_ratio = len(neg_points) / len(corrected_temp)
        neg_penalty = np.mean(np.abs(neg_points)) if len(neg_points) > 0 else 0
        
        baseline_diff2 = np.diff(baseline_temp, n=2)
        smoothness_penalty = np.sum(np.abs(baseline_diff2))

        kurtosis_score = kurtosis(corrected_temp, fisher=False) 

        lambda_scores.append({
            'lambda': current_lambda,
            'neg_ratio': neg_ratio,
            'neg_penalty': neg_penalty,
            'smoothness_penalty': smoothness_penalty,
            'kurtosis_score': kurtosis_score,
        })

    neg_penalties = [s['neg_penalty'] for s in lambda_scores]
    smoothness_penalties = [s['smoothness_penalty'] for s in lambda_scores]
    kurtosis_scores = [s['kurtosis_score'] for s in lambda_scores]
    
    min_neg_p = min(neg_penalties)
    max_neg_p = max(neg_penalties)
    min_smooth_p = min(smoothness_penalties)
    max_smooth_p = max(smoothness_penalties)
    min_kurtosis = min(kurtosis_scores) 
    max_kurtosis = max(kurtosis_scores) 

    best_lambda = lambda_candidates[0]
    best_score = float('inf')

    for score_info in lambda_scores:
        norm_neg_p = (score_info['neg_penalty'] - min_neg_p) / (max_neg_p - min_neg_p) if (max_neg_p - min_neg_p) > 0 else 0
        norm_smooth_p = (score_info['smoothness_penalty'] - min_smooth_p) / (max_smooth_p - min_smooth_p) if (max_smooth_p - min_smooth_p) > 0 else 0
        norm_kurtosis = (score_info['kurtosis_score'] - min_kurtosis) / (max_kurtosis - min_kurtosis) if (max_kurtosis - min_kurtosis) > 0 else 0

        under_subtraction_penalty = 1 - norm_kurtosis 
        
        w_neg_ratio = 1.0 
        w_neg_penalty = 1.0  
        w_smoothness = 1.0 
        w_under_subtraction = 1.0 

        score = (score_info['neg_ratio'] * w_neg_ratio + 
                 norm_neg_p * w_neg_penalty + 
                 norm_smooth_p * w_smoothness + 
                 under_subtraction_penalty * w_under_subtraction)

        score_info['total_score'] = score

        print(f"Lambda {score_info['lambda']:.0e}: 负值比例={score_info['neg_ratio']:.3f}, "
              f"负值大小={norm_neg_p:.3f}, 平滑度惩罚={norm_smooth_p:.3f}, "
              f"扣除不足惩罚={under_subtraction_penalty:.3f}, 最终评分={score:.4f}")

        if score < best_score:
            best_score = score
            best_lambda = score_info['lambda']
    
    return best_lambda, lambda_scores

def test(model, device, keys, spectrum, filename, spectrum_raw, spectrum_max,spectrum1,spectrum2,
         keys_new1,spectrum_raw1,spectrum_max1,keys_new2,spectrum_raw2,spectrum_max2,g,h,file):
    import matplotlib.pyplot as pl
    model.eval()
    start_time = time.time() 
    spectrum1 = Variable(spectrum1).to(device)
    output1 = model(spectrum1)
    output1 = np.array(output1.cpu().detach().numpy()[0, 0, :])
    spectrum1 = np.array(spectrum1.cpu().detach().numpy()[0, 0, :])
    
    x1 = keys_new1
    weights1 = output1   
    if lambda_value1 == 1e0:
        best_lambda1, _ = auto_select_lambda(spectrum_raw1, weights1, differences)
        print(f"第一段自动选择的最优lambda: {best_lambda1:.0e}")
    else:
        best_lambda1 = lambda_value1
        print(f"第一段使用手动指定lambda: {best_lambda1:.0e}")
    a5 = np.copy(weights1)
    itermax = 500

    for m in range(1, itermax + 1):
        c6 = WhittakerSmooth(spectrum_raw1, a5, best_lambda1, differences)
        d = spectrum_raw1 - c6
        neg_indices = d < 0

        if not np.any(neg_indices):
            print(f"第一段在第 {m} 次迭代时收敛，所有点均非负，提前退出。")
            break

        dssn = np.abs(np.sum(d[neg_indices]))
        if dssn < 1e-6:  
            print(f"第一段在第 {m} 次迭代时收敛，负向偏差过小，提前退出。")
            break
            
        weights_to_update = a5[neg_indices]
        deviations = np.abs(d[neg_indices])
        
        learning_rate = 1.0
        increment = 1.0 + learning_rate * (deviations / np.max(deviations))
        
        a5[neg_indices] = weights_to_update * increment

        max_weight = np.max(a5)
        a5[0] = max_weight
        a5[-1] = max_weight
    
    baseline_corrected1 = WhittakerSmooth(spectrum_raw1, a5, best_lambda1, differences)
    baseline_corrected1 = baseline_corrected1 * spectrum_max1
    
    model.eval()
    spectrum2 = Variable(spectrum2).to(device)
    output2 = model(spectrum2)
    output2 = np.array(output2.cpu().detach().numpy()[0, 0, :])
    spectrum2 = np.array(spectrum2.cpu().detach().numpy()[0, 0, :])
    
    x2 = keys_new2
    weights2 = output2   
    if lambda_value2 == 1e0:
        best_lambda2, _ = auto_select_lambda(spectrum_raw2, weights2, differences)
        print(f"第二段自动选择的最优lambda: {best_lambda2:.0e}")
    else:
        best_lambda2 = lambda_value2
        print(f"第二段使用手动指定lambda: {best_lambda2:.0e}")
    a6 = np.copy(weights2)
    
    for m in range(1, itermax + 1):
        c6 = WhittakerSmooth(spectrum_raw2, a6, best_lambda2, differences)
        d = spectrum_raw2 - c6
        
        neg_indices = d < 0
        
        if not np.any(neg_indices):
            print(f"第二段在第 {m} 次迭代时收敛，所有点均非负，提前退出。")
            break

        dssn = np.abs(np.sum(d[neg_indices]))
        if dssn < 1e-6:  
            print(f"第二段在第 {m} 次迭代时收敛，负向偏差过小，提前退出。")
            break

        weights_to_update = a6[neg_indices]
        deviations = np.abs(d[neg_indices])

        learning_rate = 1.0
        increment = 1.0 + learning_rate * (deviations / np.max(deviations))
        
        a6[neg_indices] = weights_to_update * increment

        max_weight = np.max(a6)
        a6[0] = max_weight
        a6[-1] = max_weight

    baseline_corrected2 = WhittakerSmooth(spectrum_raw2, a6, best_lambda2, differences)
    baseline_corrected2 = baseline_corrected2 * spectrum_max2
    
    a3 = [1000] * 5000
    baseline_corrected = WhittakerSmooth(np.append(baseline_corrected1, baseline_corrected2), a3, lambda_value, differences)
    f = interp1d(np.append(keys_new1, keys_new2), baseline_corrected, kind='linear')
    x = np.linspace(g, h, 2500)
    baseline_corrected = f(x)
    spectrum_raw = spectrum_raw * spectrum_max
    baseline_corrected = baseline_corrected 
    output_corrected = spectrum_raw - baseline_corrected
    
    if Plot_Switch == 1:
        fig, ax = pl.subplots(nrows=2, ncols=1, tight_layout=True, figsize=(8, 8))
        ax[0].plot(x, baseline_corrected, '-r', label='基线')
        # ax[0].plot(x1, baseline_corrected1, '-g', label='基线1')
        # ax[0].plot(x2, baseline_corrected2, '-r', label='基线2')
        ax[0].plot(x, spectrum_raw, '-b', label='原始光谱')
        ax[0].legend()
        ax[0].set_title('光谱与基线')
    
        ax[1].plot(x, output_corrected, '-b')
        ax[1].set_title('基线校正后光谱')
    
        # ax[2].plot(x1, output1, '-g', label='第一段模型输出')
        # ax[2].plot(x2, output2, '-r', label='第二段模型输出')
        # ax[2].legend()
        # ax[2].set_title('模型输出')
        # ax[3].plot(x1, a5, '-g', label='第一段最终权重')
        # ax[3].plot(x2, a6, '-r', label='第二段最终权重')
        # ax[3].legend()  
        # ax[3].set_title('迭代后的权重')
    end_time = time.time()
    processing_time = end_time - start_time
    if lambda_value1 == 1e0 or lambda_value2 == 1e0:
        pl.suptitle(f"{filename}\n第一段λ={format(best_lambda1, '.0e')}, 第二段λ={format(best_lambda2, '.0e')} 处理时间: {processing_time:.2f}s", 
                    family='SimHei')
    else:
        pl.suptitle(f"{filename}\n第一段λ={format(lambda_value1, '.0e')}, 第二段λ={format(lambda_value2, '.0e')} 处理时间: {processing_time:.2f}s", 
                    family='SimHei')
    pl.show()

    root, ext = os.path.splitext(os.path.basename(file))
    head, tail = os.path.split(file)
    if Save_Switch == 1:
        c = os.path.join(head + '/bc_sg')
        isExists = os.path.exists(c)
        if not isExists:
            os.makedirs(c)
        write(os.path.join(c, root + '_bc_sg.txt'), x, output_corrected)
    
        c2 = os.path.join(head + '/base_sg')
        isExists = os.path.exists(c2)
        if not isExists:
            os.makedirs(c2)
        write(os.path.join(c2, root + '_base_sg.txt'), x, baseline_corrected)
 

def main(N_D_DATA_TRAIN, g, h, q):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #DEVICE = torch.device('cpu')
    top_dir = N_D_DATA_TRAIN
    files = os.listdir(top_dir)
    for filename in files:
        file = os.path.join(top_dir, filename)
        file = file.replace('\\', '/')
        print('Begin:'+file)
        keys, spectrum =read(file)
        f = interp1d(keys, spectrum, kind='linear')
        keys_new = np.linspace(g, h, 2500)
        spectrum = f(keys_new)
        spectrum_raw = spectrum
        spectrum_max = max(spectrum_raw)
        spectrum_raw = normalization(spectrum_raw)
        spectrum = PEER(spectrum_raw , 5)
        for p in range(1):
            spectrum = PEER(spectrum, 5)
        spectrum = normalization(spectrum)
        spectrum = torch.tensor(spectrum)
        spectrum = spectrum.reshape(1, spectrum.shape[0])
        spectrum = spectrum.reshape(1, spectrum.shape[0], spectrum.shape[1])  
        spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
        spectrum = spectrum.permute(1, 0, 2)
        
        keys_new1 = np.linspace(g, q, 2500)  
        spectrum1 = f(keys_new1)
        # pl.plot(keys_new1,spectrum1)
        # pl.show()
        spectrum_raw1 = spectrum1
        spectrum_max1 = max(spectrum_raw1)
        spectrum_raw1 = normalization(spectrum_raw1)
        spectrum1 = PEER(spectrum_raw1 , 5)
        for p in range(3):
            spectrum1 = PEER(spectrum1, 5)
        spectrum1 = normalization(spectrum1)
        spectrum1 = torch.tensor(spectrum1)
        spectrum1 = spectrum1.reshape(1, spectrum1.shape[0])
        spectrum1 = spectrum1.reshape(1, spectrum1.shape[0], spectrum1.shape[1])  
        spectrum1= torch.as_tensor(spectrum1, dtype=torch.float32)
        spectrum1 = spectrum1.permute(1, 0, 2)
        
        keys_new2 = np.linspace(q, h, 2500)
        spectrum2 = f(keys_new2)
        # pl.plot(keys_new2,spectrum2)
        # pl.show()
        spectrum_raw2 = spectrum2
        spectrum_max2 = max(spectrum_raw2)
        spectrum_raw2 = normalization(spectrum_raw2)
        spectrum2 = PEER(spectrum_raw2 , 5)
        for p in range(3):
            spectrum2= PEER(spectrum2, 5)
        spectrum2 = normalization(spectrum2)
        spectrum2 = torch.tensor(spectrum2)
        spectrum2 = spectrum2.reshape(1, spectrum2.shape[0])
        spectrum2 = spectrum2.reshape(1, spectrum2.shape[0], spectrum2.shape[1])  
        spectrum2= torch.as_tensor(spectrum2, dtype=torch.float32)
        spectrum2 = spectrum2.permute(1, 0, 2)
        filename, ext = os.path.splitext(filename)
        model = torch.load(MODEL_PATH,weights_only=False)
        test(model, DEVICE, keys, spectrum, filename, spectrum_raw, spectrum_max,spectrum1,spectrum2,keys_new1,spectrum_raw1,spectrum_max1,keys_new2,spectrum_raw2,spectrum_max2,g,h,file)

N_D_DATA_TRAIN = r'e:\Desktop\Liu_Group Algorithm\2.data\VMS'
script_dir = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(script_dir, 'model', '0925RU_2500.pth')  
differences = 2   

lambda_value1 = 1e6  #第一段平滑度参数，值为1e0表示自动选择，其他值为手动指定
lambda_value2 = 1e8   #第二段平滑度参数，值为1e0表示自动选择，其他值为手动指定
lambda_value = 1e10   #控制两条段基线连接处的平滑度参数
Plot_Switch = 1   #是否绘图     1为绘图，0为不绘图
Save_Switch = 0   #是否保存结果  1为保存，0为不保存。默认保存至数据文件夹下的bc_sg（基线矫正）和base_sg（基线）文件夹中
main(N_D_DATA_TRAIN, 201, 2999, 2500)    #截断起点，截断终点，截断中点   分段拟合低波数和高波数
#分段拟合专用框架


