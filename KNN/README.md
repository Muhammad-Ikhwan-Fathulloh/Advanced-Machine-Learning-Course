# Sentiment Analysis dengan NLP dan KNN

- Mata Kuliah: Advance Machine Learning
- Nama: Muhammad Ikhwan Fathulloh

## 1. Instalasi dan Import Library

```bash
!pip install Sastrawi
```

```python
import pandas as pd
import numpy as np
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
```

Library yang digunakan:

Sastrawi: Untuk stemming dan stopword removal pada teks berbahasa Indonesia.
scikit-learn: Untuk preprocessing data, TF-IDF, dan algoritma KNN.
pandas & numpy: Untuk manipulasi dan analisis data.
re: Untuk cleansing teks dengan regular expressions.

## 2. Memuat Dataset

```python
url = "https://raw.githubusercontent.com/Muhammad-Ikhwan-Fathulloh/Advanced-Machine-Learning-Course/refs/heads/main/KNN/Datasets/sentiment_cellular.csv"
data = pd.read_csv(url, encoding='latin-1')
```

Dataset dimuat dari URL dalam format CSV.

## 3. Preprocessing Teks

### a. Case Folding
Mengubah semua huruf menjadi huruf kecil menggunakan fungsi lower().

```python
def casefolding(text):
    return text.lower()

data['Text Tweet'] = data['Text Tweet'].apply(casefolding)
```

### b. Cleansing
Membersihkan teks dari:
- Tanda baca
- Angka
- Karakter tunggal
- Spasi berlebih

```python
def cleansing(text):
    text = re.sub(r'[?|$|.|!_:")(-+,]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Hapus karakter tunggal
    text = re.sub('\s+', ' ', text)  # Hapus spasi berlebih
    return text.strip()

data['Text Tweet'] = data['Text Tweet'].apply(cleansing)
```

### c. Tokenization dan Stopword Removal
- Tokenisasi dilakukan dengan membagi teks menjadi kata-kata.
- Stopword (kata-kata umum yang tidak penting) dihapus menggunakan Sastrawi.

```python
def sastrawi_tokenization(text):
    text = stopword_remover.remove(text)
    return text.split()

data['Text Tweet'] = data['Text Tweet'].apply(sastrawi_tokenization)
```

### d. Stemming
Stemming digunakan untuk mengembalikan kata-kata ke bentuk dasarnya.

```python
def stemming(tokens):
    return ' '.join([stemmer.stem(token) for token in tokens])

data['Text Tweet'] = data['Text Tweet'].apply(stemming)
```

## 4. TF-IDF Vectorization
TF-IDF digunakan untuk mengonversi teks ke dalam representasi numerik.

```python
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['Text Tweet'])
```

- TF-IDF (Term Frequency-Inverse Document Frequency): Mengukur pentingnya suatu kata dalam dokumen relatif terhadap seluruh koleksi dokumen.

## 5. Split Data
Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
```

## 6. KNN Model
Model K-Nearest Neighbors (KNN) dilatih pada data pelatihan dengan jumlah tetangga (n_neighbors=7).

```python
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
```

### Prediksi dan Evaluasi
Model diuji pada data pengujian, dan metrik evaluasi seperti akurasi, presisi, recall, dan F1 score dihitung.

```python
y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
```

## 7. Menyimpan Model dan TF-IDF Vectorizer
Model yang telah dilatih dan vectorizer disimpan menggunakan joblib.

```python
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
```

## 8. Prediksi Sentimen Baru
Model yang disimpan dimuat kembali untuk melakukan prediksi pada teks baru.

```python
model = joblib.load('knn_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_text = ["Bagus nih"]
new_text_tfidf = tfidf_vectorizer.transform(new_text)
prediction = model.predict(new_text_tfidf)

print("Prediksi Sentimen:", prediction)
```