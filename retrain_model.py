import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Memuat dataset (ubah path sesuai dengan lokasi dataset Anda)
data_df = pd.read_csv('E:/Download/archive/creditcard.csv')

# Pisahkan fitur dan target
X = data_df.drop('Class', axis=1)  # Menghapus kolom 'Class' dari data fitur
y = data_df['Class']  # Kolom target 'Class'

# Pisahkan dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Terapkan SMOTE untuk mengatasi ketidakseimbangan data pada data latih
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Membangun model Random Forest setelah SMOTE
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Prediksi pada data uji
y_pred = model.predict(X_test)

# Evaluasi performa model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Cetak hasil evaluasi
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Menyimpan model yang telah dilatih ke file .pkl
joblib.dump(model, 'model.pkl')
print("Model has been saved as 'model.pkl'")