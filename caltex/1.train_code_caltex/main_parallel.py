import os
import time
import pathlib

import pandas as pd
import numpy as np
from tqdm import tqdm 
from pmdarima.arima import auto_arima
import copy
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import Parallel,delayed

from xgboost_ray import train, RayDMatrix, RayParams #ray imports
from sagemaker_ray_helper import RayHelper
import ray

from sagemaker_ray_helper import RayHelper


# ray_helper = RayHelper()
# ray_helper.start_ray()

def GSC_Model_VAR(train,h,time_seq):

    Train = copy.deepcopy(train)
    Train = Train.iloc[:time_seq]
    
    VAR_model = VAR(Train)
    try:
        AIC_list = [VAR_model.fit(i).aic for i in range(1,5)]
        min_AIC_index = AIC_list.index(min(AIC_list))
        VAR_result = VAR_model.fit(min_AIC_index+1)
        lag = VAR_result.k_ar
        results = VAR_result.forecast(Train.values[-lag:],steps=h)

        pred = []

        for i in range(0,h):
            pred.append(results[i][0])

        pred = pd.Series(pred,index=[np.arange(time_seq,time_seq+h)])
    except:
        pred = [9999, 9999, 9999]
 
    return pred, VAR_result


def GSC_Variable_Selection(df, time_step):
    ms = MinMaxScaler()
    selection_list = []
    for i in range(1,df.shape[1]):
        for j in range(time_step):
            tmp_df = ms.fit_transform(np.array(df.iloc[:,[0,i]].dropna()))*0.9 + 0.1
            tmp_df_log = np.log(pd.DataFrame(tmp_df)).diff().dropna()
            # tmp_df = np.log(pd.DataFrame(tmp_df))
            g_results = grangercausalitytests(tmp_df_log, maxlag=[j+1], verbose=False)
            p_value = g_results[j+1][0]['ssr_chi2test'][1]
            if(p_value < 0.05):
                selection_list.append(i)
                break
    return selection_list


def time_idx(df, time_seq, pred_period, shift_num):

    for i in range(len(df)-time_seq-pred_period):
        
        train = copy.deepcopy(df.iloc[i:time_seq+i+shift_num])
        valid = df.iloc[(time_seq+i):(time_seq+i+pred_period),1]
        train = train.reset_index(drop=True)

        date_seq = copy.deepcopy(train["DATE"])
        train = train.drop(["DATE"],axis=1)
        
        
        timeseries=[train, valid, date_seq]

        if i==0:
            time_idx_assignment=[[train, valid, date_seq]]

        else:
            timeseries=[train, valid, date_seq]
            time_idx_assignment.append(timeseries)
    
    return time_idx_assignment


def feature_combi(granger_list:list, combi_num:int=3):
    if len(granger_list) < combi_num:
        print("Error: 변수 개수 보다 조합에 적용되는 개수가 더 많습니다.")
    else:
        combi_result = []
        for i in range(combi_num):
            combi_list = list(itertools.combinations(granger_list,i+1))
            for combi in combi_list:
                combi_result.append([0]+list(combi))
        return combi_result


def check_sagemaker(data_path, model_dir):
    ## SageMaker
    if os.environ.get('SM_MODEL_DIR') is not None:
        data_path = os.environ['SM_CHANNEL_TRAIN']
        model_dir =os.environ.get('SM_MODEL_DIR')
    return data_path, model_dir



def main():
    print("Start Main...")
    start = time.time()
    data_path = '../data'
    model_dir = '../model'
    
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    data_path, model_dir= check_sagemaker(data_path, model_dir)
    # ray_helper = RayHelper()
    # ray_helper.start_ray()
    
    n_core = os.cpu_count()-2
    df = pd.read_csv(f"{data_path}/tmp_df.csv").dropna()

    df = df.reset_index(drop = True)
    df_granger = df.drop(["DATE"], axis = 1)
    time_step = 3
    h = 3
    
    print("Start feature selection ...")
    lo_selected = GSC_Variable_Selection(df_granger, time_step)
    combi_list = feature_combi(lo_selected, 3)        
    time_seq, pred_periods, shift_num = 36, 3, 3
    time_idx_assignment = time_idx(df,time_seq,pred_periods,shift_num)
    
    import random
    random.seed(7777)
    nums = random.sample(range(len(combi_list)), 3)

    print(f" ************* nums : {nums}")
    
    import tqdm as tqdm
    scaler = 'None'
    
    pred_result = []
    model_result = []
    
    print(f" ************* n_core : {n_core}")
    
    print("Start a training job ...")
    # for i in tqdm.tqdm(range(len(df)-time_seq-h*2)):
    #     for j in ((nums)):
    #         pred, model = GSC_Model_VAR(pd.DataFrame(time_idx_assignment[i][0].iloc[:,combi_list[j]]), h, time_seq) 
    #         pred_result.append(pred)
    #         model_result.append(model)
    
    with Parallel(n_jobs=(n_core)) as parallel:
        results = parallel(delayed(GSC_Model_VAR)(pd.DataFrame(time_idx_assignment[i][0].iloc[:,combi_list[j]]), h, time_seq)
        for j in ((nums)) for i in tqdm.tqdm(range(len(df)-time_seq-h*2)))
    
    for preds, models in results:
        pred_result.append(preds)
        model_result.append(models)
    
    print("Save the trained model")
    from statsmodels.iolib import load_pickle, save_pickle
    
    save_pickle(model_result, model_dir + '/save_model.pkl')
    
    
    taken = time.time() - start
    print(f"TRAIN TIME TAKEN: {taken:.2f} seconds")
    # with Parallel(n_jobs=(n_core)) as parallel:
    
#         globals()['{}_results'.format('VAR')] = parallel(delayed(GSC_Model_VAR)(pd.DataFrame(time_idx_assignment[i][0].iloc[:,combi_list[j]]), h, time_seq)
#         for j in ((nums)) for i in tqdm.tqdm(range(len(df)-time_seq-h*2)))


if __name__ == '__main__':
    main()