import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Cargar el conjunto de datos desde un archivo CSV
file_path = "winequality-white.csv"
df = pd.read_csv(file_path, sep=";")

# Separar las características (X) de las etiquetas (y)
X = df.drop("quality", axis=1)  # Excluir la columna de calidad
y = df["quality"]

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características (especialmente importante para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar y entrenar el modelo SVM
svm_model = SVC(kernel="linear")  # Puedes cambiar el kernel según tus necesidades (lineal, radial, polinómico, etc.)
svm_model.fit(X_train_scaled, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test_scaled)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Mostrar el informe de clasificación
print("Informe de clasificación:\n", classification_report(y_test, y_pred))
