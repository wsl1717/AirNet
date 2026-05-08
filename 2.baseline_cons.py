#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   2.baseline_cons.py
@Time    :   2025/10/14 14:53:07
@Author  :   sl.Wang
@Version :   1.0
同源框架
'''
import os
import time
import torch
import numpy as np
import copy
from torch.autograd import Variable
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from model.ResUNet_2500 import ResUNet_2500
import matplotlib.pyplot as pl
from scipy.stats import kurtosis 
from natsort import natsorted 

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

def WhittakerSmooth(x, w, lambda_, differences=2):
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)

def calculate_snr(spectrum):
    spectrum = np.array(spectrum)  
    signal = np.max(spectrum)
    noise_points_start = spectrum[:10] 
    noise_points_end = spectrum[-10:] 
    noise_points = np.concatenate((noise_points_start, noise_points_end))
    noise = np.std(noise_points)
    
    if noise > 1e-10:
        snr = signal / noise
        snr = np.log10(snr) * 20  
    else:
        snr = 1000  
    #print("信号强度 (Signal):", signal, " 噪声标准差 (Noise):", noise)
    print(f"计算得到的光谱信噪比 (SNR): {snr:.2f}dB")
    return snr

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

        if score < best_score:
            best_score = score
            best_lambda = score_info['lambda']
    
    return best_lambda, lambda_scores

def test_with_auto_lambda(model, device, keys, spectrum_PEER, filename, spectrum_raw, 
                         spectrum_max, file, x, lambda_value, differences=2):
    model.eval()
    spectrum_PEER = Variable(spectrum_PEER).to(device)
    output_model = model(spectrum_PEER)
    output_model = np.array(output_model.cpu().detach().numpy()[0, 0, :])
    spectrum_PEER = spectrum_PEER.cpu().numpy().flatten()
    
    if lambda_value == 1e0:
        weights = output_model  
        best_lambda, lambda_scores = auto_select_lambda(spectrum_PEER, weights, differences)
        #print(f"自动选择模式: 选择最优lambda={best_lambda:.0e}")

    else:
        best_lambda = lambda_value
        lambda_scores = None
        weights = output_model
        #print(f"手动指定模式: 使用lambda={best_lambda:.0e}")    
    itermax_initial = 500
    a4 = np.copy(weights)  

    for m in range(1, itermax_initial + 1):
        z = WhittakerSmooth(spectrum_PEER, a4, best_lambda, differences)
        d = spectrum_PEER - z

        neg_indices = d < 0
        
        if not np.any(neg_indices):
            #print(f"在第 {m} 次迭代时收敛，所有点均非负，提前退出。")
            break

        dssn = np.abs(np.sum(d[neg_indices]))
        if dssn < 1e-6: 
            #print(f"在第 {m} 次迭代时收敛，负向偏差过小，提前退出。")
            break
            
        weights_to_update = a4[neg_indices]
        deviations = np.abs(d[neg_indices])
        
        learning_rate = 1 
        increment = 1.0 + learning_rate * (deviations / np.max(deviations))
        
        a4[neg_indices] = weights_to_update * increment

        max_weight = np.max(a4)
        a4[0] = max_weight
        a4[-1] = max_weight
         
    baseline_corrected = WhittakerSmooth(spectrum_PEER, a4, best_lambda, differences)
    spectrum_PEER = spectrum_PEER * spectrum_max
    baseline_corrected = baseline_corrected * spectrum_max
    output_corrected = spectrum_PEER - baseline_corrected
      
    return best_lambda, lambda_scores, output_corrected

def test_with_auto_lambda_unified(model, device, keys, spectrum_PEER, filename, spectrum_raw, 
                                spectrum_max, file, x, unified_lambda, unified_weights, differences=2):
    start_time = time.time() 
    model.eval()
    spectrum_PEER = Variable(spectrum_PEER).to(device)
    output_model = model(spectrum_PEER)
    output_model = np.array(output_model.cpu().detach().numpy()[0, 0, :])
    spectrum_PEER = spectrum_PEER.cpu().numpy().flatten()
    
    print(f"使用统一参数: lambda={unified_lambda:.0e}")

    itermax_unified = 10
    a4 = np.copy(unified_weights)
    
    for m in range(1, itermax_unified + 1):
        z = WhittakerSmooth(spectrum_PEER, a4, unified_lambda, differences)
        d = spectrum_PEER - z
        neg_indices = d < 0
        
        if not np.any(neg_indices):
            print(f"在第 {m} 次迭代时收敛，所有点均非负，提前退出。")
            break
        dssn = np.abs(np.sum(d[neg_indices]))
        if dssn < 1e-6:
            print(f"在第 {m} 次迭代时收敛，负向偏差过小，提前退出。")
            break
            
        weights_to_update = a4[neg_indices]
        deviations = np.abs(d[neg_indices])      
        learning_rate = 1 
        increment = 1.0 + learning_rate * (deviations / np.max(deviations))      
        a4[neg_indices] = weights_to_update * increment

        max_weight = np.max(a4)
        a4[0] = max_weight
        a4[-1] = max_weight
         
    baseline_corrected = WhittakerSmooth(spectrum_PEER, unified_weights, unified_lambda, differences)
    spectrum_PEER = spectrum_PEER * spectrum_max
    baseline_corrected = baseline_corrected * spectrum_max
    output_corrected = spectrum_PEER - baseline_corrected

    if Plot_Switch == 1:
        fig, ax = pl.subplots(nrows=3, ncols=1, tight_layout=True, figsize=(8, 8))
        ax[0].plot(x, spectrum_raw * spectrum_max,'-m', label='原始光谱')
        ax[0].plot(x, spectrum_PEER,'-b', label='PEER去噪')
        ax[0].legend()
        ax[0].set_title('PEER去噪效果')      
        ax[1].plot(x, spectrum_PEER, '-b', label='PEER去噪')   
        ax[1].plot(x, baseline_corrected, '-r', label='基线')
        ax[1].legend()
        ax[1].set_title('光谱与基线') 
        ax[2].plot(x, output_corrected, '-b')
        ax[2].set_title('基线矫正后的光谱')
        
        end_time = time.time()
        processing_time = end_time - start_time       
        pl.suptitle(f"{filename} \n 统一λ={format(unified_lambda, '.0e')}  处理时间: {processing_time:.2f}s", 
                    family='SimHei')
        pl.show()
    
    root, ext = os.path.splitext(os.path.basename(file))
    head, tail = os.path.split(file)
    if Save_Switch == 1:
        c1 = os.path.join(head + '/bc_cons')
        isExists = os.path.exists(c1)
        if not isExists:
            os.makedirs(c1)
        write(os.path.join(c1, root + '_bc_cs.txt'), x, output_corrected)   
        # 保存基线
        c2 = os.path.join(head + '/base_cons')
        isExists = os.path.exists(c2)
        if not isExists:
            os.makedirs(c2)
        write(os.path.join(c2, root + '_base_cs.txt'), x, baseline_corrected)
    
    return unified_lambda

def main(N_D_DATA_TRAIN, g, h):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")   
    top_dir = N_D_DATA_TRAIN
    files = os.listdir(top_dir)
    files=natsorted(files) 
    model = torch.load(MODEL_PATH, weights_only=False, map_location=DEVICE)
    
    b = np.zeros(2500)  
    snr_z = 0           
    q = len(files)     
    all_spectra_data = [] 
    
    for filename in files:
        print(f"第一阶段分析文件: {filename}")
        file = os.path.join(top_dir, filename).replace('\\', '/')
        if os.path.isfile(file) and file.endswith("txt"):
            try:
                keys, spectrum = read(file)
                f = interp1d(keys, spectrum, kind='linear')
                keys_new = np.linspace(g, h, 2500)
                spectrum_raw = f(keys_new)
                spectrum_max = max(spectrum_raw)
                spectrum_raw = normalization(spectrum_raw)
                
                for p in range(5):
                    spectrum_PEER = PEER(spectrum_raw.tolist(), c=7)

                spectrum_tensor = torch.tensor(spectrum_PEER)
                spectrum_tensor = spectrum_tensor.reshape(1, spectrum_tensor.shape[0])
                spectrum_tensor = spectrum_tensor.reshape(1, spectrum_tensor.shape[0], spectrum_tensor.shape[1])
                spectrum_tensor = torch.as_tensor(spectrum_tensor, dtype=torch.float32)
                spectrum_tensor = spectrum_tensor.permute(1, 0, 2)
                
                spectrum_tensor_gpu = Variable(spectrum_tensor).to(DEVICE)
                
                _, _, corrected_temp = test_with_auto_lambda(
                    model, DEVICE, keys, spectrum_tensor, 
                    os.path.splitext(filename)[0], spectrum_raw, 
                    spectrum_max, file, keys_new, 
                    lambda_value, differences
                )
                #corrected_temp_norm = corrected_temp / spectrum_max
                corrected_temp_norm = corrected_temp 
                snr = calculate_snr(corrected_temp_norm)
                
                output_model = model(spectrum_tensor_gpu)
                a4 = np.array(output_model.cpu().detach().numpy()[0, 0, :])

                weighting_factor = abs(snr) 
                b = b + (a4 * weighting_factor)
                snr_z = snr_z + weighting_factor
                
                all_spectra_data.append({
                    'filename': filename,
                    'file': file,
                    'keys': keys,
                    'keys_new': keys_new,
                    'spectrum_raw': spectrum_raw,
                    'spectrum_PEER': spectrum_PEER,
                    'spectrum_max': spectrum_max,
                    'spectrum_tensor': spectrum_tensor,
                    'snr': snr
                })
                
            except Exception as e:
                print(f"第一阶段分析文件 {filename} 时出错: {str(e)}")
                continue
    
    #unified_weights = b / (q * snr_z)  
    unified_weights = b / snr_z 
    
    print(f"统一权重计算完成。总SNR: {snr_z:.2f}dB, 样本数量: {q}")

    if lambda_value == 1e0:
        print("自动选择统一Lambda...")
        best_spectrum = max(all_spectra_data, key=lambda x: x['snr'])
        print(f"使用信噪比最高的光谱 '{best_spectrum['filename']}' (SNR: {best_spectrum['snr']:.2f}) 来确定统一Lambda")
        unified_lambda, _ = auto_select_lambda(best_spectrum['spectrum_PEER'], unified_weights, differences)
        print(f"确定的统一Lambda: {unified_lambda:.0e}")
    else:
        unified_lambda = lambda_value
        print(f"使用手动指定的Lambda: {unified_lambda:.0e}")
    
    for data in all_spectra_data:
        print(f"\n第二阶段处理文件: {data['filename']}")
        
        unified_lambda = test_with_auto_lambda_unified(
            model, DEVICE, data['keys'], data['spectrum_tensor'], 
            os.path.splitext(data['filename'])[0], data['spectrum_raw'], 
            data['spectrum_max'], data['file'], data['keys_new'], 
            unified_lambda, unified_weights, differences
        )

DATA_PATH = r'e:\Desktop\Liu_Group Algorithm\2.data\single spectra'   
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'model', '0925RU_2500.pth')
differences = 2

"过度扣除基线的时候，增大平滑度参数。扣除不到位的时候，减小平滑度参数。"
lambda_value = 1e0 #平滑度  值为1e0表示自动选择平滑度，如果自动选择的平滑度不合适，可以修改这个参数
Plot_Switch = 1  #是否绘图     1为绘图，0为不绘图
Save_Switch = 1  #是否保存结果  1为保存，0为不保存。默认保存至数据文件夹下的bc_cons（基线矫正）和base_cons（基线）文件夹中
main(DATA_PATH, 2981, 3786)  #起始波数,终止波数  两侧尽量多留出一些背景信息
#同源数据专用框架
