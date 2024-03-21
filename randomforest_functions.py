def processamento(file):
    import pandas as pd
    
    
    dados = pd.read_csv(file)
    return dados, dados.info(), dados['Target'].unique(), dados[['Idade na matrícula', 'Taxa de desemprego', 'Taxa de inflação', 'PIB']].describe()

def plot_analises(dados):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    
    #Gráficos Demográficos
    sns.displot(dados['Idade na matrícula'], bins=20)
    plt.show()
    color_dict = {'Desistente': 'red', 'Graduado': 'green', 'Matriculado': 'blue'}
    sns.set_palette(list(color_dict.values())) #Padronizando como paleta de cores
    sns.displot(data=dados, x='Idade na matrícula', hue='Target', kind='kde', fill=True)
    plt.show() #Distribuição normal dos alunos em relação a sua situação.
    sns.countplot(x='Sexo', hue='Target', data=dados)
    plt.show()
    
    #Gráfico Econômicos
    color_dict = {'Desistente': 'red', 'Graduado': 'green', 'Matriculado': 'blue'}
    sns.countplot(x='Devedor', hue='Target', data=dados)
    plt.show()
    sns.countplot(x='Taxas de matrícula em dia', hue='Target', data=dados)
    plt.show()
    sns.countplot(x='Bolsista', hue='Target', data=dados)
    plt.show()
    sns.boxenplot(x='Target', y='disciplinas 1º semestre (notas)', data=dados)
    plt.show()
    contagem = dados.groupby(['Curso', 'Target']).size().reset_index(name='Contagem')
    contagem['Porcentagem'] = contagem.groupby('Curso')['Contagem'].transform(lambda x: (x/x.sum())*100)
    fig = px.bar(contagem, y='Curso', x='Porcentagem', color='Target', orientation='h',
                 color_discrete_map=color_dict)
    fig.show()
    
    
def encoder(dados):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    
    
    colunas_categoricas = ['Estado civil', 'Migração', 'Sexo', 'Estrangeiro', 'Necessidades educacionais especiais', 'Devedor', 
                       'Taxas de matrícula em dia', 'Bolsista', 'Curso', 'Período', 'Qualificação prévia']
    encoder = OneHotEncoder(drop='if_binary')
    dados_categoricos = dados[colunas_categoricas]
    dados_encoded = pd.DataFrame(encoder.fit_transform(dados_categoricos).toarray(), 
                             columns=encoder.get_feature_names_out(colunas_categoricas))

    dados_final = pd.concat([dados.drop(colunas_categoricas, axis=1), dados_encoded], axis=1)
    return dados_final

def feature_engineering(dados_final):
    from sklearn.model_selection import train_test_split
    
    
    x = dados_final.drop(['Target'], axis=1)
    y = dados_final['Target']

    x, x_test, y, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=0)
    x_treino, x_val, y_treino, y_val  = train_test_split(x, y, stratify=y, random_state=0)
    
    return x_treino, y_treino, x_test, y_test, x_val, y_val

def applying_model(x_treino, y_treino, x_test, y_test, x_val, y_val):
    from sklearn.ensemble import RandomForestClassifier
    
    
    rfmodel1 = RandomForestClassifier(random_state=0, max_depth=5)
    rfmodel1.fit(x_treino, y_treino)
    y_predval = rfmodel1.predict(x_val)
    print(f'Acurácia de Treino:{rfmodel1.score(x_treino, y_treino)}')
    print(f'Acurácia de Validação:{rfmodel1.score(x_val, y_val)}')
    y_predtest = rfmodel1.predict(x_test)
    
    return y_predval, y_predtest, print(f'Acurácia de Treino:{rfmodel1.score(x_treino, y_treino)}'), print(f'Acurácia de Validação:{rfmodel1.score(x_val, y_val)}')

def metrics(y_val, predict_validation, y_test, predict_test):
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    confusion_matrix_validation = confusion_matrix(y_val, predict_validation)
    confusion_matrix_test = confusion_matrix(y_test, predict_test)
    
    accuracy = accuracy_score(y_test, predict_test)
    precision = precision_score(y_test, predict_test, average='weighted')
    recall = recall_score(y_test, predict_test, average='weighted')  
    f1 = f1_score(y_test, predict_test, average='weighted')
    
    target_names = ['Desistente', 'Graduando', 'Matriculado']
    report = classification_report(y_test, predict_test, target_names=target_names)
    
    
    return confusion_matrix_validation, confusion_matrix_test, accuracy, precision, recall, f1, report