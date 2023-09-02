import numpy as np
import pandas as pd
import ace_lib as ace
import helpful_functions as hf
import random
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
import numpy as np
np.set_printoptions(suppress=True)
# warnings.filterwarnings("ignore")

class GeneticAlgorithm():
    def __init__(self, father_num=30, gene_num=3,
                       son_num=10, crossoverrate=0.9, 
                       mutationrate=0.1, maxiteration=5 ):
        self.F = father_num
        self.G = gene_num
        self.S = son_num
        self.cr = crossoverrate
        self.mr = mutationrate
        self.max_iter = maxiteration
    
    def generatefather(self, df, num_sampindatafield=200, num_ingeneration=200 ): # num_ingeneration is the output of father
        num_sampindatafield = num_sampindatafield
        self.F = num_ingeneration
        father_expression_list = [] # generate population
        operator_m = [ 'ts_scale', 'ts_rank' ]
        for x in random.sample(df.id.values.tolist(), num_sampindatafield):  # 從每個資料框中隨機抽樣 id 值
            for m in operator_m:
                for y in ['5', '21', '63', '126']:
                    expression = f"{m}( {x}, {y} )"
                    father_expression_list.append(expression)
        if len(father_expression_list) <= self.F:
            random.sample(father_expression_list, self.F)  #deap
        else:
            return father_expression_list
        return father_expression_list
    
    def alphagenerate(generation_expression_list, df):
        Father_alpha_list = [ace.generate_alpha( x, region="USA", universe="TOP3000", neutralization="SUBINDUSTRY",) for x in generation_expression_list]
        return Father_alpha_list

    def datasafe(df):
        save_path = rf'C:\Users\user\ACE2023_v3\database\{df}.csv'  # 使用 f-string 插入变量值
        df.to_csv(save_path, index=False)
        print(rf'{df} already save in : {save_path}')
    
    def fitnessfilter(df, turnoverlimite = 0.8, fitnessrank = 10):
        data = df
        turnoverlimite = turnoverlimite
        fitnessrank = fitnessrank 
        filtered_data = data[data['turnover'] < turnoverlimite] # select the suitable turnover "0.7"
        survive_fitness_data = filtered_data.nlargest( fitnessrank , 'fitness')  # select the greatest alpha "10"
        return survive_fitness_data

    def crossover(self, df, crossoverrate = 0.9):
        self.cr = crossoverrate
        beforecrossover_expression_list = df['expression'].tolist()
        son_expression_list = [] # new generation_expression_list
        parent1 = expression1.split(' ') 
        parent2 = expression2.split(' ')
        for i in range(0, len(beforecrossover_expression_list), 2):
            expression1 = beforecrossover_expression_list[i]
            expression2 = beforecrossover_expression_list[i + 1]
            if random.random() < self.cr: # crossover as "0.9"
                docrossover_expression1 = f"{parent1[0]} {parent2[1]} {parent1[2]}" # new born son
                son_expression_list.append(docrossover_expression1)
                docrossover_expression2 = f"{parent2[0]} {parent1[1]} {parent2[2]}"
                son_expression_list.append(docrossover_expression2)
            else:
                son_expression_list.append(expression1)
                son_expression_list.append(expression2)
        return son_expression_list

    def mutation(self, expression_list, mutationrate=0.1):
        self.mr = mutationrate
        new_generation_expression_list = []
        for i in range(len(expression_list)):
            if random.random() < self.mr:
                nbmutedgene = expression_list[i] # nb means "need to be" 
                nbmutedgene = nbmutedgene.split(' ')
                mutagene_expression = self.generatefather(df = expression_list ,num_sampindatafield = 1, num_ingeneration = 1)
                mutagene = mutagene_expression.split(' ')
                if random.random() < 0.5:
                    muted_expression = f"{mutagene[0]} {nbmutedgene[1]} {mutagene[2]}"
                    new_generation_expression_list.append(muted_expression)
                else:
                    muted_expression = f"{nbmutedgene[0]} {mutagene[1]} {nbmutedgene[2]}"
                    new_generation_expression_list.append(muted_expression)
            else:
                nhmutedgene = expression_list[i] # nh means nothing happen
                new_generation_expression_list.append(nhmutedgene)

    def main(self, df):
        print('father_num :', self.F, 
              'gene_num :', self.G, 
              'son_num :', self.S,
              'crossoverrate :', self.cr,
              'mutationrate :', self.mr,
              'maxiteration :', self.max_iter)
        it = 0
        while it < self.max_iter: # where the loop start
            father_expression_list = self.generatefather(df ,num_sampindatafield = 200, num_ingeneration = 200)
            if it >= 1:
                father_expression_list.append(newgeneration_list)
            else:
                pass
            father_alpha_df_list = self.alphagenerate(father_expression_list)
            father_alpha_df = pd.DataFrame(father_alpha_df_list)
            father_alpha_df_csv= self.datasafe(father_alpha_df)
            result = ace.simulate_alpha_list_multi(s, father_alpha_df)
            len(result)
            result[0].keys()
            result[0]['is_stats']
            result_performance = hf.prettify_result(result, detailed_tests_view=False)
            result_performance.head()
            result_performance_csv = self.datasafe(result_performance) # show and safe the simulation result
        
            survive_fitness_data = self.fitnessfilter(result_performance) # fitnessfuntion
            survive_fitness_data_csv = self.datasafe(survive_fitness_data)

            best_fitness_iteration = [] # iterate graphic
            best_fitness_iteration.append(survive_fitness_data['fitness'][0]) 

            son_expression_list = self.crossover(df,crossoverrate = 0.9) # crossover
        
            mutagene_list = self.generatefather(df, num_sampindatafield = 1, num_ingeneration = len(son_expression_list) ) # It's wierd because this should be at ga.mutation
            newgeneration_list = self.mutation(son_expression_list, mutagene_list, mutationrate=0.1)
            newgeneration_list_csv = self.datasafe(newgeneration_list)
            
            it += 1
    
        print('The best 10 fitness: ', min(survive_fitness_data))
        best_index = survive_fitness_data.index(max(fitness))
        print('Setup list is: ')
        print(survive_fitness_data[best_index])
        plt.figure(figsize = (15,8))
        plt.xlabel("Iteration",fontsize = 15)
        plt.ylabel("Fitness",fontsize = 15)
        plt.plot(best_fitness_iteration,linewidth = 2, label = "Best fitness convergence", color = 'b')
        plt.legend()
        plt.show()
    
    if __name__ == '__main__':
        main() 

