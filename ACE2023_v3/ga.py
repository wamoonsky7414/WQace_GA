import ace_lib as ace # !pip install first!! & may delete the matplot library
import helpful_functions as hf
import pandas as pd
import requests
import plotly.express as px
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
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

df = combined_df
def generategeneration(df, num_sampindatafield=50, num_ingeneration=50 ): # num_ingeneration is the output of father
    expression_list = [] # generate population
    operator_m = [ 'ts_scale', 'ts_rank' ]
    for x in random.sample(df.id.values.tolist(), num_sampindatafield):  # 從每個資料框中隨機抽樣 id 值
        for m in operator_m:
            for y in ['5', '21', '63', '126']:
                expression = f"{m}( {x}, {y} )"
                expression_list.append(expression)
    if len(expression_list) <= num_ingeneration:
        random.sample(expression_list, num_ingeneration)  #deap
    else:
        return expression_list
    
def alphagenerate(expression_list, df):
    Father_alpha_list = [ace.generate_alpha( x, region="USA", universe="TOP3000", neutralization="SUBINDUSTRY",) for x in expression_list]
    return Father_alpha_list

def datasafe(df):
    save_path = rf'C:\Users\user\ACE2023_v3\database\{df}.csv'  # 使用 f-string 插入变量值
    df.to_csv(save_path, index=False)
    print(rf'{df} already save in : {save_path}')
    
def fitnessfilter(df, turnoverlimite = 0.8, fitnessrank = 10):
    fitnessrank = fitnessrank 
    filtered_data = df[df['turnover'] < turnoverlimite] # select the suitable turnover "0.7"
    survive_fitness_data = filtered_data.nlargest( fitnessrank , 'fitness')  # select the greatest alpha "10"
    return survive_fitness_data

def crossover(df, crossoverrate = 0.9):
    bfcrossover_expression_list = df['expression'].tolist() # bf means before
    atcrossover_expression_list = [] # at means after
    parent1 = expression1.split(' ') 
    parent2 = expression2.split(' ')
    for i in range(0, len(bfcrossover_expression_list), 2):
        expression1 = bfcrossover_expression_list[i]
        expression2 = bfcrossover_expression_list[i + 1]
        if random.random() < crossoverrate: 
            docrossover_expression1 = f"{parent1[0]} {parent2[1]} {parent1[2]}" # new born son
            atcrossover_expression_list.append(docrossover_expression1)
            docrossover_expression2 = f"{parent2[0]} {parent1[1]} {parent2[2]}"
            atcrossover_expression_list.append(docrossover_expression2)
        else:
            atcrossover_expression_list.append(expression1)
            atcrossover_expression_list.append(expression2)
    return atcrossover_expression_list

def mutation(expression_list, mutationrate=0.1):
    new_generation_expression_list = []
    for i in range(len(expression_list)):
        if random.random() < mutationrate:
            bfmutedgene = expression_list[i] 
            bfmutedgene = bfmutedgene.split(' ')
            mutagene_expression = generategeneration(df = expression_list ,num_sampindatafield = 1, num_ingeneration = 1)
            mutagene = mutagene_expression.split(' ')
            trigger = random.random()
            if trigger < 0.333:
                muted_expression = f"{mutagene[0]} {bfmutedgene[1]} {bfmutedgene[2]}"
                new_generation_expression_list.append(muted_expression)
            elif  0.333 <= trigger < 0.666:
                muted_expression = f"{bfmutedgene[0]} {mutagene[1]} {bfmutedgene[2]}"
                new_generation_expression_list.append(muted_expression)                
            else :
                muted_expression = f"{bfmutedgene[0]} {bfmutedgene[1]} {mutagene[2]}"
                new_generation_expression_list.append(muted_expression)
        else:
            nhmutedgene = expression_list[i] # nh means nothing happen
            new_generation_expression_list.append(nhmutedgene)

def iteration_grahic(df):
    survive_fitness_data = df
    best_fitness_iteration = [] # iterate graphic
    best_fitness_iteration.append(survive_fitness_data['fitness'][0]) 

it = 0
    parent_expression_list = generategeneration(df ,num_sampindatafield = 200, num_ingeneration = 200)

while it <= 100: # where the loop start
    parent_alpha_df_list = alphagenerate(parent_expression_list)
    parent_alpha_df = pd.DataFrame(parent_alpha_df_list)
    parent_alpha_df_csv= datasafe(parent_alpha_df)
    result = ace.simulate_alpha_list_multi(s, parent_alpha_df)

    result_performance = hf.prettify_result(result, detailed_tests_view=False)
    result_performance_csv = datasafe(result_performance) # show and safe the simulation result
        
    survive_fitness_data = fitnessfilter(result_performance) # fitnessfuntion + select
    survive_fitness_data_csv = datasafe(survive_fitness_data)
    iteration_grah_list = iteration_grahic(survive_fitness_data) # not ready to print yet see line 150~
    if it == 100:
        break

    son_expression_list = crossover(df,crossoverrate = 0.9) # crossover
        
    mutagene_list = generategeneration(df, num_sampindatafield = 1, num_ingeneration = len(son_expression_list) ) 
    newgeneration_list = mutation(son_expression_list, mutagene_list, mutationrate=0.1)
    newgeneration_list_csv = datasafe(newgeneration_list)
            
    it += 1
    

print('The best 10 fitness: ')
print(survive_fitness_data)
plt.figure(figsize = (15,8))
plt.xlabel("Iteration",fontsize = 15)
plt.ylabel("Fitness",fontsize = 15)
plt.plot(iteration_grah_list,linewidth = 2, label = "Best fitness convergence", color = 'b')
plt.legend()
plt.show()
 
