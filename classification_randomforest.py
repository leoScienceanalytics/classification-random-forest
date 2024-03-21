from randomforest_functions import processamento, plot_analises_demograficas, plot_analises_economicas, encoder
from randomforest_functions import feature_engineering, applying_model
from sklearn.metrics import confusion_matrix, roc_curve

file = 'dados_estudantes.csv'

dados, dados_info, target_unicos, descibre_numericos = processamento(file) #Processamento dos dados
analises_demo = plot_analises_demograficas(dados) #Gráfico Demográficos
print(analises_demo)
analises_eco = plot_analises_economicas(dados) #Gráficos Econômicos
print(analises_eco)
dados_final = encoder(dados)
x_treino, y_treino, x_test, y_test, x_val, y_val = feature_engineering(dados_final)
predict_validation, predict_test, accuracy_train, accuracy_validation = applying_model(x_treino, y_treino, x_test, y_test, x_val, y_val)

print(predict_test)
print(predict_validation)
print(accuracy_train)
print(accuracy_validation)

cm = confusion_matrix(y_val, predict_validation)
print(cm)

cm1 = confusion_matrix(y_test, predict_test)
print(cm1)