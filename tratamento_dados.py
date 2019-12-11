import numpy as np
import pandas as pd
from sklearn import preprocessing

def main():
        rdo_1 = pd.read_csv("crime-data-in-brazil/RDO_1.csv", low_memory = False)
        rdo_2 = pd.read_csv("crime-data-in-brazil/RDO_2.csv", low_memory = False)
        rdo_3 = pd.read_csv("crime-data-in-brazil/RDO_3.csv", low_memory = False)

        #dropping extra column
        rdo_1.drop('Unnamed: 30', axis = 1, inplace = True)
        rdo_2.drop('Unnamed: 30', axis = 1, inplace = True)
        rdo_3.drop('Unnamed: 30', axis = 1, inplace = True)

        frames = [rdo_1, rdo_2, rdo_3]
        data = pd.concat(frames, ignore_index=True)

        data['IDADE_PESSOA'] = pd.to_numeric(data['IDADE_PESSOA'], errors='coerce')
        data['NUMERO_LOGRADOURO'] = pd.to_numeric(data['NUMERO_LOGRADOURO'], errors='coerce')

        data['LONGITUDE'] = pd.to_numeric(data['LONGITUDE'], errors='coerce')
        data['LATITUDE'] = pd.to_numeric(data['LATITUDE'], errors='coerce')

        data['DATA_OCORRENCIA_BO'] = pd.to_datetime(data['DATA_OCORRENCIA_BO'], errors='coerce')
        data['HORA_OCORRENCIA_BO'] = pd.to_datetime(data['HORA_OCORRENCIA_BO'], format='%H:%M', errors='coerce').dt.time

        data['CIDADE'] = pd.Categorical(data['CIDADE'])
        data['NOME_SECCIONAL_CIRC'] = pd.Categorical(data['NOME_SECCIONAL_CIRC'])
        data['NOME_DELEGACIA_CIRC'] = pd.Categorical(data['NOME_DELEGACIA_CIRC'])
        data['DESCR_TIPO_BO'] = pd.Categorical(data['DESCR_TIPO_BO'])
        data['RUBRICA'] = pd.Categorical(data['RUBRICA'])
        data['DESCR_CONDUTA'] = pd.Categorical(data['DESCR_CONDUTA'])
        data['DESDOBRAMENTO'] = pd.Categorical(data['DESDOBRAMENTO'])
        data['DESCR_TIPOLOCAL'] = pd.Categorical(data['DESCR_TIPOLOCAL'])
        data['DESCR_SUBTIPOLOCAL'] = pd.Categorical(data['DESCR_SUBTIPOLOCAL'])
        data['LOGRADOURO'] = pd.Categorical(data['LOGRADOURO'])
        data['DESCR_TIPO_PESSOA'] = pd.Categorical(data['DESCR_TIPO_PESSOA'])
        data['SEXO_PESSOA'] = pd.Categorical(data['SEXO_PESSOA'])
        data['COR_CUTIS'] = pd.Categorical(data['COR_CUTIS'])   

        bool = (data['FLAG_VITIMA_FATAL'] != 'N') & (data['FLAG_VITIMA_FATAL'] != 'S')
        data['FLAG_VITIMA_FATAL'].loc[bool] = np.NaN

        bool = (data['SEXO_PESSOA'] != 'M') & (data['SEXO_PESSOA'] != 'F')
        data['SEXO_PESSOA'].loc[bool] = np.NaN

        #'Preta' 'Parda' 'Branca' 'Amarela' 'Outros' 'Vermelha'
        data['COR_CUTIS'] = data['COR_CUTIS'].str.strip()
        bool = (data['COR_CUTIS'] != 'Preta') & (data['COR_CUTIS'] != 'Parda') & (data['COR_CUTIS'] != 'Branca') & (data['COR_CUTIS'] != 'Amarela') & (data['COR_CUTIS'] != 'Outros') & (data['COR_CUTIS'] != 'Vermelha')
        data['COR_CUTIS'].loc[bool] = np.NaN  

        bool_values = data['FLAG_VITIMA_FATAL'].notnull()

        crimes = data[bool_values]
        crimes.drop('ID_DELEGACIA', axis = 1, inplace = True)
        crimes.drop('NOME_DEPARTAMENTO', axis = 1, inplace = True)
        crimes.drop('NOME_SECCIONAL', axis = 1, inplace = True)
        crimes.drop('NOME_DELEGACIA', axis = 1, inplace = True)
        crimes.drop('NUM_BO', axis = 1, inplace = True)
        crimes.drop('NOME_DEPARTAMENTO_CIRC', axis = 1, inplace = True)
        crimes.drop('NOME_MUNICIPIO_CIRC', axis = 1, inplace = True)
        crimes.drop('DATAHORA_COMUNICACAO_BO', axis = 1, inplace = True)
        crimes.drop('NUMERO_LOGRADOURO', axis = 1, inplace = True)
        crimes.drop('DESCR_CONDUTA', axis = 1, inplace = True)
        crimes.drop('DESDOBRAMENTO', axis = 1, inplace = True)
        crimes.drop('DESCR_SUBTIPOLOCAL', axis = 1, inplace = True)
        crimes.drop('LOGRADOURO', axis = 1, inplace = True)

        dmCidade = pd.get_dummies(crimes['CIDADE'])
        dmNomeSceccional = pd.get_dummies(crimes['NOME_SECCIONAL_CIRC'])
        dmNomeDelegacia = pd.get_dummies(crimes['NOME_DELEGACIA_CIRC'])
        dmDescBO = pd.get_dummies(crimes['DESCR_TIPO_BO'])
        dmRubrica = pd.get_dummies(crimes['RUBRICA'])
        dmTipoLocal = pd.get_dummies(crimes['DESCR_TIPOLOCAL'])
        dmTipoPessoa = pd.get_dummies(crimes['DESCR_TIPO_PESSOA'])
        dmSexoPessoa = pd.get_dummies(crimes['SEXO_PESSOA'])
        dmCor = pd.get_dummies(crimes['COR_CUTIS'])

        vitima = pd.concat([vitima, dmCidade, dmNomeSceccional, dmDescBO, dmRubrica,
                    dmTipoLocal, dmTipoPessoa, dmSexoPessoa, dmCor], axis=1)

        crimes.drop('CIDADE', axis = 1, inplace = True)
        crimes.drop('NOME_SECCIONAL_CIRC', axis = 1, inplace = True)
        crimes.drop('NOME_DELEGACIA_CIRC', axis = 1, inplace = True)
        crimes.drop('DESCR_TIPO_BO', axis = 1, inplace = True)
        crimes.drop('RUBRICA', axis = 1, inplace = True)
        crimes.drop('DESCR_TIPOLOCAL', axis = 1, inplace = True)
        crimes.drop('DESCR_TIPO_PESSOA', axis = 1, inplace = True)
        crimes.drop('SEXO_PESSOA', axis = 1, inplace = True)
        crimes.drop('COR_CUTIS', axis = 1, inplace = True)

        enc = preprocessing.OrdinalEncoder()

        crimes['FLAG_STATUS'] = enc.fit_transform(crimes['FLAG_STATUS'].values.reshape(-1, 1))
        crimes['FLAG_VITIMA_FATAL'] = enc.fit_transform(crimes['FLAG_VITIMA_FATAL'].values.reshape(-1, 1))

        bool_values = crimes['HORA_OCORRENCIA_BO'].notnull()
        vitima = crimes[bool_values]
        bool_values = crimes['LATITUDE'].notnull()
        vitima = crimes[bool_values]
        bool_values = crimes['IDADE_PESSOA'].notnull()
        vitima = crimes[bool_values]

        crimes.reset_index(inplace=True, drop = True)

        list = pd.Series.tolist(crimes['DATA_OCORRENCIA_BO'])

        months = []
        days = []
        for item in list:
                months.append(item.month)
                days.append(item.day)

        list = pd.Series.tolist(crimes['HORA_OCORRENCIA_BO'])

        hours = []
        for item in list:
                hours.append(item.hour)

        d = {'MES': months, 'DIA': days, 'HORA': hours}
        df = pd.DataFrame(d)

        vitima = pd.concat([vitima,df], axis = 1, sort=False)

        crimes.drop('DATA_OCORRENCIA_BO', axis = 1, inplace = True)
        crimes.drop('HORA_OCORRENCIA_BO', axis = 1, inplace = True)

        crimes.to_csv('dados.csv')

if __name__ == '__main__':
    main()