import numpy as np
from numpy.random import seed
        
def mse(y):
    y_predicao = y.mean()
    y_atual = y
    return np.mean(np.power(y_predicao - y_atual, 2))

class Tree():
    def __init__(self, data, min_sample=1):
        self.no_raiz = Node(np.ones(data.shape[0], dtype='bool'))
        self.limite_minimo_por_node = min_sample
        self.splits=[]
        
    def criterio_parada(self, y):
        if (len(set(y)) <= 1) | (len(y) <= self.limite_minimo_por_node) :
            return True
        else:
            return False

    def busca_melhor_corte(self, data, y, no_pai_idxs, features_ativas, valores_unicos):
        max_ir = -np.inf
        m, n = data.shape
        idxs_list = []
        melhor_corte_idxs = []
        # Se não houver possibilidade de corte, esses são os valores iniciais
        melhor_corte = (-1,0)
        num_instancias_no_pai = len(y[no_pai_idxs])
        idxs = np.arange(num_instancias_no_pai)
        y_pai = 1.*np.bincount(y[no_pai_idxs], minlength = self.n_classes)/num_instancias_no_pai
        mse_pai = mse(y_pai)
        idxs_features = np.arange(n)[features_ativas]

        for j in idxs_features:
            # Se a coluna atual não houver mais possibilidade de corte, pula pra próxima
            if len(set(data[no_pai_idxs, j])) == 1:
                features_ativas[j] = True
                continue
            else:
            # A coluna tual ainda pode existir cortes
                values = np.intersect1d(valores_unicos[j], data[no_pai_idxs, j])
                for ponto_medio in values[:-1]:
                    # Vetor de indices para o nó filho esquerdo
                    idxs_list = [(data[:,j] <= ponto_medio), (data[:,j] > ponto_medio)]
                    idxs_list[0] = (idxs_list[0] & no_pai_idxs)
                    idxs_list[1] = (idxs_list[1] & no_pai_idxs)
                    
                    num_instacias_no_filho = np.array([len(y[instances]) for instances in idxs_list])
                    # Calcula a proporção de instâncias do filho esquerdo e direito 
                    # em relação ao número de instâncias do nó pai
                    proporcao_nos_filhos = 1.*num_instacias_no_filho/num_instancias_no_pai
            
                    lista_entropia = np.array([mse(y[instances]) for instances in idxs_list])
                    ir = mse_pai - np.dot(proporcao_nos_filhos, lista_entropia)
                    
                    # Se o último máximo for menor que a redução de impureza atual 
                    # e os nós filhos esquerdo e direito tiverem mais de limite_minimo_por_node, 
                    # as instâncias atualizam a melhor divisão
                    if (max_ir < ir) & np.all(num_instacias_no_filho > self.limite_minimo_por_node):
                        # Atualiza para o melhor máximo
                        max_ir = ir
                        # Atualiza a melhor divisão
                        melhor_corte = (j, ponto_medio)
                        # Atualiza os indices dos melhores nós esquerdo e direito
                        melhor_corte_idxs = idxs_list 
                        
        if melhor_corte[0] != -1:
            self.splits.append(melhor_corte)
        return melhor_corte, melhor_corte_idxs, features_ativas
        
    def fazer_predicao(self,data):
        m, n = data.shape
        preds = np.zeros((m, self.n_classes))
        for i in np.arange(m):
            no_raiz = self.no_raiz
            while no_raiz.terminal == False:
                split = no_raiz.get_split()
                # avalia o vetor de entrada atual para ver se usamos a divisão esquerda ou direita
                if type(data[i, split[0]]) == 'str':
                    no_raiz = no_raiz.nos_filhos(split[1])
                else:
                    if data[i, split[0]] <= split[1]:
                        no_raiz = no_raiz.nos_filhos[0]
                    else:
                        no_raiz = no_raiz.nos_filhos[1]
                
            preds[i,:] = no_raiz.get_pred()
        return preds

    def get_valores_unicos(self, data):
        m, n = data.shape
        lista_valores_unicos_features = []
        mtx = np.zeros(n, dtype='bool') # Vetor com mais de um valor por feature
        for j in np.arange(n):
            lista_valores_unicos_features.append(np.unique(data[:,j]))
            if len(lista_valores_unicos_features[j]) > 1:
                # Se existir mais de um valor único, então é uma feature_ativa
                mtx[j] = True
        self.no_raiz.set_features_ativas(mtx)
        return lista_valores_unicos_features
        
    def criar(self, data, y):
        self.n_classes = 1
        valores_unicos_features = self.get_valores_unicos(data)
        lista_nos = [self.no_raiz]
        while lista_nos:
            ultimo_no = lista_nos.pop()
            if self.criterio_parada(y[ultimo_no.idxs]):
                # Se obedecer o critério de parada, é um nó terminal e é calculado a predição
                ultimo_no.set_as_terminal()
                num_instances = len(y[ultimo_no.idxs])
                pred = np.mean(y[ultimo_no.idxs])
                ultimo_no.set_pred(pred)
            else:
                split, idxs_list, features_ativas = \
                    self.busca_melhor_corte(data, \
                                            y, \
                                            ultimo_no.idxs, \
                                            ultimo_no.get_features_ativas(), \
                                            valores_unicos_features)
                ultimo_no.set_features_ativas(features_ativas)
                if split[0] != -1:
                    # Se não obedeceu o criterio de parada
                    # E ainda houver possibilidade de cortes,
                    # Então cria os nós filhos
                    nos_filhos = [Node(idxs_list[i]) for i in range(len(idxs_list))]
                    for filho in nos_filhos:
                        filho.set_features_ativas(features_ativas) 
                    ultimo_no.set_split(split, nos_filhos)
                    lista_nos.extend(nos_filhos)
                else:
                    # Se não houver mais possibilisdades de cortes, 
                    # é setado o nó atual como terminal e calculado a predição
                    ultimo_no.set_as_terminal()
                    num_instances = len(y[ultimo_no.idxs])
                    pred = np.mean(y[ultimo_no.idxs])
                    ultimo_no.set_pred(pred)

class Node():
    def __init__(self, idxs):
        self.nos_filhos = []
        self.idxs = idxs
        self.pred = 0
        self.split = ()
        self.terminal = False
        self.features_ativas = [] #features com mais possiblidade de divisão

    def set_features_ativas(self, feats):
        self.features_ativas = feats

    def get_features_ativas(self):
        return self.features_ativas

    def set_nos_filhos(self, nos_filhos):
        self.nos_filhos = nos_filhos

    def set_split(self, split, nos_filhos):
        self.split = split
        self.set_nos_filhos(nos_filhos)

    def get_split(self):
        return self.split

    def set_pred(self, pred):
        self.pred = pred

    def set_as_terminal(self):
        self.terminal = True

    def get_pred(self):
        return self.pred

    def show_tree(self, depth = 0, cond = ' '):
        base = '    ' * depth + cond
        # Leaf
        #print(base + 'if X[' + 'FEATURE_NAME' + '] <= ' + str(self.pred))
        print(self.get_split)
        for no in self.nos_filhos:
            no.show_tree(depth + 1, 'then ')
        #self.right.show_tree(depth + 1, 'else ')