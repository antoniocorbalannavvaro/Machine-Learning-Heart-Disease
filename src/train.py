import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

#Por si no leyera el csv:
# os.chdir("")

#Cargamos el Dataset:
df = pd.read_csv("data\processed\Heart_limpio.csv")


#Separamos la variable tarjet del resto de datos:
X = df[['ST_Slope', 'Oldpeak', "Cholesterol", 'Age', "RestingBP", 'MaxHR',"Sex"]]

y = df["HeartDisease"]


#Separamos los datos en Test y Train:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 2)


#Creamos y entrenamos el modelo:
estimator = DecisionTreeClassifier(max_depth=60,random_state=56)

modelo = BaggingClassifier(
    base_estimator = estimator,
    n_estimators=50,
    max_samples=300,
    bootstrap=True,
    max_features = 6,
    random_state=1)

modelo.fit(X_train, y_train)

#Comprobamos la precisión:
y_pred = modelo.predict(X_test)
accuracy_score(y_test, y_pred)


#Exportamos modelo:
joblib.dump(modelo, "new_model.pkl")










