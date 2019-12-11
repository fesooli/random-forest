from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from time import *
from math import sqrt
from tree import Tree
import matplotlib.pyplot as plt

def criar_arvore(data, y):
        idxs = np.arange(data.shape[0])
        instances = np.random.choice(idxs,data.shape[0],replace=True)
        arvore = Tree(data[instances,:])
        arvore.criar(data[instances,:],y[instances])
        yield arvore

def fazer_predicao(data, trees):
        m, n = data.shape
        pred = np.zeros((m, trees[0].n_classes))
        for tree in trees:
            pred += tree.fazer_predicao(data)
        pred = 1.*pred/len(trees)
        return pred
       
def erro_quadratico_medio(y_test, predicao):
        somatorio = 0
        n = len(y_test) 
        for i in range (0,n): 
                diferenca = y_test[i] - predicao[i] 
                quadrado_diferenca = diferenca ** 2   
                somatorio = somatorio + quadrado_diferenca 
        return somatorio/n 

def main():
        tempo_inicial = time()
        crimes = pd.read_csv("dados.csv", low_memory = False)

        crimes["FLAG_STATUS"] = crimes["FLAG_STATUS"].astype(int)
        crimes["IDADE_PESSOA"] = crimes["IDADE_PESSOA"].astype(int)
        crimes['FLAG_VITIMA_FATAL'] = crimes['FLAG_VITIMA_FATAL'].astype(int)
        y = crimes['FLAG_VITIMA_FATAL']
        X = crimes.drop('FLAG_VITIMA_FATAL', axis = 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        numero_arvores = [1,2,3,4,5,6,7,8,9,10]
        sc = SparkSession.builder \
                .master("local") \
                .appName("Floresta Aleatória") \
                .getOrCreate() \
                .sparkContext
        rdd = sc.parallelize(numero_arvores)

        preds = sc.parallelize([rdd \
                .flatMap(lambda x: criar_arvore(X_train.values, y_train.values)).collect()]) \
                .flatMap(lambda v: fazer_predicao(X_test.values, v))\
                .flatMap(lambda x: x).collect()

        arvore = list(criar_arvore(X_train.values, y_train.values))[0]
        preds_arvore = arvore.fazer_predicao(X_test.values)
        preds_unica = []
        for p in preds_arvore:
                preds_unica.append(p[0])

        # Salvar csv de comparação da predição
        data = {"BASE": y_test.values, \
                "PREVISTO_RANDOM_FOREST": preds, \
                "PREVISTO_DECISION_TREE": preds_unica}
        predicao_final = pd.DataFrame(data, columns = ['BASE', \
                                                        'PREVISTO_RANDOM_FOREST', \
                                                        'PREVISTO_DECISION_TREE'])

        largura_barra = 10
        plt.figure(figsize=(10,5))
        r1 = np.arange(len(y_test.values))
        r2 = [x + largura_barra for x in r1]
        r3 = [x + largura_barra for x in r2]

        plt.bar(r1, y_test.values, color='red', width=largura_barra, label='BASE')
        plt.bar(r2, preds, color='yellow', width=largura_barra, label='PREVISTO_RANDOM_FOREST')
        plt.bar(r3, preds_unica, color='blue', width=largura_barra, label='PREVISTO_DECISION_TREE')

        plt.xlabel('Linhas')
        plt.ylabel('Predição')
        plt.title('Comparação de dados da Base x Random Forest x Decision Tree')
        plt.legend()
        plt.show()

        predicao_final.to_csv('random_forest.csv')

        erro_medio = erro_quadratico_medio(y_test.values, preds)
        print('Erro Quadratico Médio:', erro_medio)
        print('Raiz Erro Quadratico Médio:', np.sqrt(erro_medio))

        erro_medio = erro_quadratico_medio(y_test.values, preds_unica)
        print('Erro Quadratico Médio Árvore:', erro_medio)
        print('Raiz Erro Quadratico Médio Árvore:', np.sqrt(erro_medio))

        tempo_final = time()
        tempo_gasto = tempo_final - tempo_inicial
        print("Tempo do algoritmo: %.3f segundos" % tempo_gasto)

if __name__ == '__main__':
        main()