import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

#Por si no leyera el csv:
# os.chdir("")

#Cargamos el Dataset:
df = pd.read_csv("data\processed\Heart_limpio.csv")


#Separamos la variable tarjet del resto de datos:
X = df[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']]

y = df["HeartDisease"]


#Separamos los datos en Test y Train:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


#Escalamos los datos: (0,1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Creamos y entrenamos el modelo:
modelo = KNeighborsClassifier(n_neighbors=9)
modelo.fit(X_train, y_train)


#Comprobamos la precisión:
print("Accuracy train", modelo.score(X_test, y_test))


#Exportamos modelo:
joblib.dump(modelo, "new_model.pkl")