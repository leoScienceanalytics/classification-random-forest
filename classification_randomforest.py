from randomforest_functions import processamento, plot_analises, encoder
from randomforest_functions import feature_engineering, applying_model, metrics


file = 'dados_estudantes.csv'

dados, dados_info, target_unicos, descibre_numericos = processamento(file) #Processamento dos dados
analises_demo = plot_analises(dados) #Gráfico Demográficos
dados_final = encoder(dados)
x_treino, y_treino, x_test, y_test, x_val, y_val = feature_engineering(dados_final)
y_predval, y_predtest, train_accuracy, val_accuracy = applying_model(x_treino, y_treino, x_test, y_test, x_val, y_val)
cm_val, cm_test, accuracy, precision, recall, f1, report = metrics(y_val, y_predval, y_test, y_predtest)
print(cm_val) #Matriz de Confusão do conjunto de validação
print(cm_test) #Matriz de Confusão do conjunto de teste
print('Acurácia:', accuracy)
print('Precisão:', precision)
print('Sensibilidade:', recall)
print('F1-score:', f1)
print(report)