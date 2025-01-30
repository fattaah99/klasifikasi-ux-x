import streamlit as st
import h5py
import pickle
import numpy as np

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
svc, tfidf_vectorizer, label_encoder = load_models("Twiter_Review_Classifier.h5")

# # Judul aplikasi
# st.title("Klasifikasi Masalah Penggunaan Aplikasi X")
# st.write("**Aplikasi ini digunakan untuk memprediksi sentimen ulasan dari Google Play Store menggunakan algoritma SVM.**")

# HTML tambahan untuk styling dan elemen visual
st.markdown(
    """
    <style>
    * {
        font-family: Arial, Helvetica, sans-serif;
    }
    nav {
        display: flex;
        padding: 5px;
        text-decoration: none;
        background-color: #3a1078;
        font-size: 20px;
    }
    nav ul {
        list-style-type: none;
        display: flex;
    }
    nav ul li a {
        text-decoration: none;
        color: white;
        margin-top: 5px;
    }
    main {
        
    }
    </style>
    <header>
      <nav>
        <ul>
          <li><a href="#">Prediksi</a></li>
        </ul>
      </nav>
    </header>
    <main>
      <div><h3>Selamat Datang di Web Klasifikasi Masalah Penggunaan Aplikasi X Berdasarkan Ulasan Google Play Store Menggunakan Algoritma SVM</h3></div>
      <p>Web ini digunakan sebagai user interface melakukan prediksi terhadap ulasan</p>
    </main>
    """,
    unsafe_allow_html=True
)

# Input ulasan dari pengguna
review_text = st.text_area("Masukkan ulasan di bawah ini:", placeholder="Ketikkan ulasan Anda di sini...")


if st.button("Prediksi"):
    if review_text.strip():
        # Transformasi teks dan prediksi
        transformed_text = tfidf_vectorizer.transform([review_text])
        prediction = svc.predict(transformed_text)
        sentiment_label = label_encoder.inverse_transform(prediction)[0]

        # Dapatkan probabilitas dari tiap kelas
        probabilities = svc.predict_proba(transformed_text)

        # Ambil probabilitas tertinggi
        max_proba = np.max(probabilities)

        # Label kelas dengan probabilitasnya
        sentiment_label = label_encoder.inverse_transform(prediction)[0]

        # Tampilkan hasil prediksi dan probabilitas
        st.subheader("Hasil Prediksi:")
        st.write(f"**Ulasan:** {review_text}")
        st.write(f"**Kategori Masalah:** {sentiment_label}")
        st.write(f"**Probabilitas:** {max_proba:.2%}")  # Menampilkan dalam persen

    else:
        st.error("Harap masukkan ulasan terlebih dahulu.")
