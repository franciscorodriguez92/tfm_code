#%% Coeficiente kappa 20% etiquetado

import pandas as pd 
from sklearn.metrics import cohen_kappa_score
import numpy as np

aurora_data = pd.read_csv("./resources/data/Aurora_etiquetas_agreement - copia.csv", header = None, names = 'c')
aurora = aurora_data['c'].tolist()

fran_data = pd.read_csv("./resources/data/Fran_etiquetas_agreement - copia.csv", header = None, names = 'c')
fran = fran_data['c'].tolist()

jhoni_data = pd.read_csv("./resources/data/Jhoni_etiquetas_agreement - copia.csv", header = None, names = 'c')
jhoni = jhoni_data['c'].tolist()

jf = cohen_kappa_score(jhoni, fran)
print('Kappa Etiquetadores 1 - 2: ' + str(jf))
jf_accuracy = len([i for i, j in zip(jhoni, fran) if i == j])/float(len(fran))
print('Tasa de coincidencia etiquetadores 1 - 2 (%): ' + str(jf_accuracy*100))

ja = cohen_kappa_score(jhoni, aurora)
print('Kappa Etiquetadores 1 - 3: ' + str(ja))
ja_accuracy = len([i for i, j in zip(aurora, jhoni) if i == j])/float(len(fran))
print('Tasa de coincidencia etiquetadores 1 - 3 (%): ' + str(ja_accuracy*100))

fa = cohen_kappa_score(fran, aurora)
print('Kappa Etiquetadores 2 - 3: ' + str(fa))
fa_accuracy = len([i for i, j in zip(aurora, fran) if i == j])/float(len(fran))
print('Tasa de coincidencia etiquetadores 2 - 3 (%): ' + str(fa_accuracy*100))


jfa_accuracy = len([i for i, j, k in zip(jhoni, fran, aurora) if i == j == k])/float(len(fran))
print('Tasa de coincidencia etiquetadores 1 - 2 - 3 (%): ' + str(jfa_accuracy*100))

print('Media Kappa Etiquetadores 1 - 2 - 3: ' + str(np.mean(np.array([jf,fa,ja]))))

#1-Jhoni
#2-Fran
#3-Aurora

