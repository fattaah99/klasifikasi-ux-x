from flask import Flask, request, render_template
import h5py
import pickle
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)


# Fungsi untuk memuat model, TF-IDF Vectorizer, dan LabelEncoder dari file .h5
def load_models(h5_path: str):
    with h5py.File(h5_path, 'r') as h5f:
        svc_data = h5f['svc_model'][()]
        svc = pickle.loads(svc_data.tobytes())
        vectorizer_data = h5f['tfidf_vectorizer'][()]
        tfidf_vectorizer = pickle.loads(vectorizer_data.tobytes())
        label_encoder_data = h5f['label_encoder'][()]
        label_encoder = pickle.loads(label_encoder_data.tobytes())
    return svc, tfidf_vectorizer, label_encoder


# Panggil fungsi untuk memuat model dan alat bantu
svc, tfidf_vectorizer, label_encoder = load_models(
    "Twiter_Review_Classifier.h5")

# Debugging: Pastikan variabel berhasil dimuat
print("Model SVC:", svc)
print("TF-IDF Vectorizer:", tfidf_vectorizer)
print("Label Encoder:", label_encoder)

# Debugging: Pastikan TF-IDF Vectorizer memiliki idf_
try:
    print("idf_ dari TF-IDF Vectorizer:", tfidf_vectorizer.idf_)
except AttributeError:
    print(
        "Error: TF-IDF Vectorizer tidak memiliki atribut idf_. Model mungkin tidak dilatih dengan benar."
    )


# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')


# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        review_text = request.form['review']

        # Transformasi teks dan prediksi
        transformed_text = tfidf_vectorizer.transform([review_text])
        prediction = svc.predict(transformed_text)
        sentiment_label = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html',
                               review=review_text,
                               sentiment=sentiment_label)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Gunakan port 5000 jika variabel PORT tidak ada
    app.run(host='0.0.0.0', port=port, debug=True)
