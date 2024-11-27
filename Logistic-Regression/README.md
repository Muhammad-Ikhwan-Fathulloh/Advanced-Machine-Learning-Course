# Prediksi Diabetes dengan Logistic Regression

- Mata Kuliah: Advance Machine Learning
- Nama: Muhammad Ikhwan Fathulloh

## 1. Instalasi dan Import Library

```python
!pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
```

Pada bagian ini, kita menginstal scikit-learn dan mengimpor berbagai library yang diperlukan untuk analisis data dan visualisasi. Beberapa di antaranya adalah pandas untuk manipulasi data, matplotlib dan seaborn untuk visualisasi, serta plotly untuk grafik interaktif.

## 2. Memuat dan Memeriksa Dataset

```python
url = "https://raw.githubusercontent.com/Muhammad-Ikhwan-Fathulloh/Advanced-Machine-Learning-Course/refs/heads/main/Logistic-Regression/Datasets/diabetes.csv"
data = pd.read_csv(url, encoding='latin-1')
df = pd.DataFrame(data)
df.info()
```

Dataset diabetes dimuat dari URL yang diberikan dan diubah menjadi DataFrame menggunakan pandas. Kita juga memeriksa informasi mengenai kolom dan tipe data menggunakan df.info().


## 3. Menangani Data yang Hilang dan Duplikat

```python
df.isnull().sum()
df.duplicated().sum()
```

Di sini, kita memeriksa adanya data yang hilang (NaN) dan duplikat dalam dataset. Hal ini penting untuk memastikan kualitas data yang akan digunakan dalam model.

## 4. Visualisasi Distribusi Data

```python
fig, ax = plt.subplots(3,3,figsize=(15,9))
for i, col in enumerate(df):
    sns.histplot(df[col], kde=True, ax=ax[i//3, i%3])
plt.show()
```

Untuk memahami distribusi fitur dalam dataset, kita menggunakan seaborn untuk membuat histogram dan menambahkan garis Kernel Density Estimate (KDE) untuk setiap kolom dalam dataset.

## 5. Mengatasi Nilai Nol (Zero) dalam Beberapa Kolom

```python
zero_col = ['Glucose','Insulin','SkinThickness','BloodPressure','BMI']
df1[zero_col] = df1[zero_col].replace(0, np.nan)
```

Beberapa kolom dalam dataset memiliki nilai 0 yang tidak valid. Kita mengganti nilai-nilai tersebut dengan NaN agar dapat ditangani lebih lanjut.

## 6. Mengisi Data yang Hilang

```python
for col in ['Glucose','Insulin','SkinThickness']:
    median_col = np.median(df1[df1[col].notna()][col])
    df1[col] = df1[col].fillna(median_col)
for col in ['BMI','BloodPressure']:
    mean_col = np.mean(df1[df1[col].notna()][col])
    df1[col] = df1[col].fillna(mean_col)
```

Setelah mengganti nilai nol dengan NaN, kita mengisi nilai yang hilang dengan nilai median untuk kolom yang mengandung nilai nol sebelumnya dan nilai rata-rata untuk kolom lainnya.

## 7. Visualisasi Korelasi Fitur

```python
sns.heatmap(data.corr(), annot=True, cmap='Reds')
plt.show()
```

Untuk memahami hubungan antara fitur, kita membuat heatmap korelasi yang menunjukkan seberapa kuat hubungan antara fitur-fitur dalam dataset.

## 8. Preprocessing Data dan Normalisasi

```python
scaler = StandardScaler()
x_norm = scaler.fit_transform(x)
x = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
```

Di bagian ini, kita melakukan normalisasi pada fitur menggunakan StandardScaler untuk memastikan bahwa setiap fitur memiliki skala yang sama dan membantu model bekerja lebih baik.

## 9. Pembagian Data ke dalam Set Pelatihan dan Pengujian

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```

Data dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan train_test_split.

## 10. Fungsi untuk Melakukan Cross Validation

```python
def Perform_cross_val(model, k, x, y, scoring):
    kf = KFold(n_splits=k)
    cv_results = cross_val_score(model, x, y, cv=kf, scoring=scoring)
    cv_mean = np.mean(cv_results)
    print(f"CV mean: {cv_mean}")
    print(f"CV results: {cv_results}\n")
```

Fungsi ini digunakan untuk melakukan validasi silang dengan KFold dan menampilkan hasil dari skor evaluasi.

## 11. Fungsi untuk Menampilkan Matriks Kebingunguan (Confusion Matrix)

```python
def plot_confusion_matrix2(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(classes)
    plt.yticks(classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
```

Fungsi ini digunakan untuk menampilkan matriks kebingunguan (confusion matrix) yang menggambarkan kinerja model dalam memprediksi kelas positif dan negatif.

## 12. Membuat Model Logistic Regression dan Melakukan Evaluasi

```python
logreg = LogisticRegression(solver='liblinear', penalty='l2', C=1)
logreg.fit(x_train, y_train)
```

Model Logistic Regression dibuat dan dilatih menggunakan data pelatihan. Model ini digunakan untuk memprediksi apakah seseorang memiliki diabetes berdasarkan fitur yang ada.

## 13. Hasil Evaluasi Model

```python
y_pred = logreg.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: {cm}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")
```

Setelah model dilatih, kita melakukan prediksi pada data pengujian dan mengevaluasi kinerjanya dengan menggunakan matriks kebingunguan dan laporan klasifikasi.

## 14. Visualisasi Hasil Model

```python
plot_confusion_matrix2(cm, classes=['Diabetes', 'No Diabetes'])
```

Matriks kebingunguan divisualisasikan untuk memberikan gambaran yang lebih jelas mengenai performa model dalam memprediksi kelas.