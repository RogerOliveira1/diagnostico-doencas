from flask import Flask, request, jsonify 
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


app = Flask(__name__)

dados = pd.read_csv('dataset-hemograma.csv')

dados = dados.fillna(dados.mean())

dados.replace([np.inf, -np.inf], np.nan, inplace=True)
dados.dropna(inplace=True)

x = dados[['Eritrocitos', 'Hemoglobina', 'Hematocrito','HCM', 'VGM', 'CHGM', 'Metarrubricitos', 'ProteinaPlasmatica', 'Leucocitos', 'Leucograma', 'Segmentados', 'Bastonetes', 'Blastos', 'Metamielocitos', 'Mielocitos', 'Linfocitos', 'Monocitos', 'Eosinofilos', 'Basofilos', 'Plaquetas']]
y = dados['Diagnostico']

model_diag = LinearRegression()
model = model_diag.fit(x, y)

resultados_regressao = {
    1: "DRC",
    2: "Hipercolesterolemia",
    3: "Animia",
    4: "Lesao Hepatica",
    5: "Infeccao Bacteriana",
    6: "Desidratacao",
    7: "Infeccao Parasitaria",
    8: "Infeccao Viral",
    9: "DRC e Cardiopatia",
    10: "Neoplasia hepatica",
    11: "Animia Hemolitica",
    12: "Diabetes",
    13: "Pre-Diabetes",
    14: "Animia e Infeccao",
    15: "Hepatopata",
    16: "Trombocitopenia e Inflamacao",
    17: "Hipoplasia Mieloide", 
    18: "Pancreatite",
    19: "Inflamacao Grave"
}

@app.route('/diagnostico', methods=['POST'])
def prever_Diagnostico():
    data = request.json

    features = np.array([[data['Eritrocitos'], data['Hemoglobina'], data['Hematocrito'], data['HCM'], data['VGM'], data['CHGM'], data['Metarrubricitos'], data['ProteinaPlasmatica'], data['Leucocitos'], data['Leucograma'], data['Segmentados'], data['Bastonetes'], data['Blastos'], data['Metamielocitos'], data['Mielocitos'], data['Linfocitos'], data['Monocitos'], data['Eosinofilos'], data['Basofilos'], data['Plaquetas']]])

    predicted_Diagnostico = model.predict(features)
    prever_Diagnostico = predicted_Diagnostico[0]

    nome_da_doenca = resultados_regressao.get(int(prever_Diagnostico), "Doença Desconhecida")

    print(f"Previsão numérica: {prever_Diagnostico}")
    print(f"Doença identificada: {nome_da_doenca}")
    
    prever_Diagnostico = round(prever_Diagnostico, 1)
    
    return jsonify({
        'prever_Diagnostico': prever_Diagnostico,
        'nome_da_doenca': nome_da_doenca
    })

if __name__ == '__main__':
    app.run(debug=True)