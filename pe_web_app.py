# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 08:31:44 2023

@author: pc
"""

import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 

# Library untuk mengabaikan warnings
import warnings
warnings.filterwarnings('ignore')


# Load save model yang telah dibuat sebelumnya
df = pd.read_csv('F:/Bismillah Skripsi/Dataset/dataset_ibu_hamil.csv')
df['level_risiko'].replace({"Tinggi": "3", "Sedang": "2", "Rendah" : "1"}, inplace=True)
df["level_risiko"] = df["level_risiko"].astype("int64")

# Memisahkan variabel independen dan dependen
x = df.drop (columns="level_risiko", axis=1)
y = df['level_risiko']

# Lakukan standarisasi untuk normalisasi data
scaler = StandardScaler()
scaler.fit(x)         
standarized_data = scaler.transform(x)

#Masukan hasil standarisasi data ke variabel X
X = standarized_data
Y = df['level_risiko']

# Buat model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

tree = DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=13)
tree.fit(X_train, Y_train)


# Membuat fungsi untuk melakukan klasifikasi level risiko

def preeklamsia_risk_level(input_data):  
    
    # Ubah data yang diinput menjadi array
    input_data_as_numpy_array = np.array(input_data)

    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshape)

    prediction =  tree.predict(std_data)

    # ubah class hasil prediksi menjadi interger agar dapat dibaca oleh model
    predicted_class = int(prediction[0])

    if predicted_class == 1:
        return 'Anda memiliki resiko rendah untuk terkena preeklamsia'
    elif predicted_class == 2:
        return 'Anda memiliki resiko sedang untuk terkena preeklamsia'
    else:
        return 'Anda memiliki resiko tinggi untuk terkena preeklamsia'
    
    

# Navigasi sidebar
with st.sidebar:
    
    selected = option_menu('Sistem Klasifikasi Level Risiko Preeklamsia',
                           
                           ['How to Use',
                            'Classification System',
                            'About'],
                           
                           icons = ['house-heart-fill', 'activity', 'person-fill'],
                           
                           default_index = 0)
    
# Halaman tata cara penggunaan
if (selected == 'How to Use'):
    
    # Judul Halaman
    st.title('Tata Cara Penggunaan Sistem Klasifikasi')
    
    
# Halaman klasifikasi level risiko
if (selected == 'Classification System'):
    
    # Judul Halaman
    st.title('Klasifikasi Level Risiko Preeklamsia')
    
    # Membagi kolom
    col1, col2 = st.columns(2)
    
    with col1:
        tinggi_badan = st.text_input('Tinggi Badan')
    
    with col2:
        berat_badan = st.text_input('Berat Badan')
    
    with col1:
        tekanan_darah_sistolik = st.text_input('Tekanan Darah Sistolik')
    
    with col2:
        tekanan_darah_diastolik = st.text_input('Tekanan Darah Diastolik')
    
    with col1:
        usia = st.text_input('Usia')
    
    with col2:
        paritas = st.text_input('Jumlah Kelahiran')
    
    with col1:
        riwayat_hipertensi = st.text_input('Riwayat Hipertensi')
    
    with col2:
        riwayat_preeklamsia = st.text_input('Riwayat Preeklamsia')
        
    # Model ML
    
    # Kode untuk prediksi
    prediksi = ''
    
    # Membuat tombol untuk klasifikasi level risiko
    if st.button('Klasifikasi Level Risiko'):
        prediksi = preeklamsia_risk_level([tinggi_badan, berat_badan, tekanan_darah_sistolik, tekanan_darah_diastolik, usia, paritas, riwayat_hipertensi, riwayat_preeklamsia])
    
    st.success(prediksi)
    

# Halaman tentang web app    
if (selected == 'About'):
    
    # Judul Halaman
    st.title('Tentang Sistem Klasifikasi Level Risiko') 

    
    
