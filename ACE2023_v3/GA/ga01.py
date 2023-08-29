import ace_lib as ace # !pip install first!! & may delete the matplot library
import helpful_functions as hf
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import random
import math
from tqdm.notebook import tqdm
import warnings
from geneticalgorithm import GeneticAlgorithm as ga
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

s = ace.start_session()

datasets_df = hf.get_datasets(s) # by default we load all datasets USA TOP3000 delay 1

# select needed datasets and become datadield
selected_datasets_df = datasets_df.query("""
        delay == 1 &\
        0.8 < coverage <= 1 &\
        0 < fieldCount < 10000 &\
        region == 'USA' &\
        universe == 'TOP3000' &\
        0 < userCount < 1000 &\
        1 < valueScore < 10 &\
        name.str.contains('volatility', case=False) == 1 
    """, engine='python').sort_values(by=['valueScore'], ascending=False)
selected_datasets_df

datafields_df_list = []  # 創建一個空列表來存儲 datafields 資料
id_list = len(selected_datasets_df.id.values.tolist())
for i in range(id_list):
    dataset_id = selected_datasets_df.id.values.tolist()[i]
    datafields_df = hf.get_datafields(s, dataset_id=dataset_id)  # 下載該資料集的所有字段
    datafields_df_list.append(datafields_df)  # 將 datafields 資料附加到列表中
    datafields_df.head()

for i, df in enumerate(datafields_df_list): # 顯示整個 datafields_df_list 中各個資料框的行數
    print("id in datafields_df", i, ":", df.shape[0])

combined_df = pd.concat(datafields_df_list, ignore_index=True) # 將所有資料框整合成一個
print("id in combined_df:", combined_df.shape[0]) # 顯示整合後的資料框行數
combined_df.head()

it = 0
while it < ga.max_iter:# where the loop start
    father_expression_list = ga.generatefather(df = combined_df ,num_sampindatafield = 200, num_ingeneration = 200)
    if it >= 1:
        merge(father_expression_list, newgeneration_list) # the one thing ledt of the all code of GA
    else:
        return father_expression_list
    father_alpha_df_list = ga.alphagenerate(father_expression_list)
    father_alpha_df = pd.DataFrame(father_alpha_df_list)
    father_alpha_df_csv= ga.datasafe(father_alpha_df)
    
    result = ace.simulate_alpha_list_multi(s, father_alpha_df)
    len(result)
    result[0].keys()
    result[0]['is_stats']
    result_performance = hf.prettify_result(result, detailed_tests_view=False)
    result_performance.head()
    result_performance_csv = ga.datasafe(result_performance) # show and safe the simulation result
    
    survive_fitness_data = ga.fitnessfilter(result_performance) # fitnessfuntion
    survive_fitness_data_csv = ga.datasafe(survive_fitness_data)
    son_expression_list = ga.crossover(df,crossoverrate = 0.9) # crossover

    mutagene_list = ga.generatefather(df = combined_df ,num_sampindatafield = 1, num_ingeneration = len(son_expression_list) ) # It's wierd because this should be at ga.mutation
    newgeneration_list = ga.mutation(son_expression_list, mutagene_list, mutationrate=0.1)
    newgeneration_list_csv = ga.datasafe(newgeneration_list)
    
    it += 1
