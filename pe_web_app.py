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
df = pd.read_csv('dataset_ibu_hamil.csv')
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

def get_value(val,my_dict):

          for key ,value in my_dict.items():

            if val == key:

              return value

def preeklamsia_risk_level(input_data):  
    
    # Ubah data yang diinput menjadi array
    input_data_as_numpy_array = np.array(input_data)

    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.transform(input_data_reshape)

    prediction =  tree.predict(std_data)

    # ubah class hasil prediksi menjadi interger agar dapat dibaca oleh model
    predicted_class = int(prediction[0])

    if predicted_class == 1:
        return 'Anda **tidak berisiko** untuk terkena preeklamsia'
    elif predicted_class == 2:
        return 'Anda memiliki **risiko sedang** untuk terkena preeklamsia'
    else:
        return 'Anda memiliki **risiko tinggi** untuk terkena preeklamsia'
    
    

# Navigasi sidebar
with st.sidebar:
    
    selected = option_menu('Deteksi Dini Preeklamsia',
                           
                           ['Cara Penggunaan',
                            'Deteksi Dini',
                            'Tentang Sistem'],
                           
                           icons = ['house-heart-fill', 'activity', 'person-fill'],
                           
                           default_index = 0)
    
# Halaman tata cara penggunaan
if (selected == 'Cara Penggunaan'):
    
    # Judul Halaman
    st.title('Tata Cara Penggunaan Sistem Klasifikasi')
    
    
# Halaman klasifikasi level risiko
if (selected == 'Deteksi Dini'):
    
    # Judul Halaman
    st.title('Deteksi Dini Risiko Preeklamsia')
    
    # Membagi kolom
    col1, col2 = st.columns(2)
    
    with col1:
        tinggi_badan = st.number_input('Tinggi Badan (cm)', min_value=100.0, max_value=240.0, value=None)
    
    with col2:
        berat_badan = st.number_input('Berat Badan (kg)', min_value=30.0, max_value=200.0, value=None)
    
    with col1:
        tekanan_darah_sistolik = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=50, max_value=240, value=None)
    
    with col2:
        tekanan_darah_diastolik = st.number_input('Tekanan Darah Diastolik (mmHg)', min_value=30, max_value=240, value=None)
    
    with col1:
        usia = st.number_input('Usia (tahun)', min_value=15, max_value=70, value=None)
    
    with col2:
        paritas = st.number_input('Jumlah Kelahiran Hidup', min_value=0, max_value=10, value=None)
    
    with col1:
        hipertensi_options = {'Pernah': 1, 'Tidak Pernah': 0}
        hipertensi = st.selectbox('Pernah mengalami tekanan darah tinggi ?', tuple(hipertensi_options.keys()))
        riwayat_hipertensi = get_value(hipertensi,hipertensi_options)
    
    with col2:
        preeklamsia_options = {'Pernah': 1, 'Tidak Pernah': 0}
        preeklamsia = st.selectbox('Pernah mengalami preeklamsia ?', tuple(preeklamsia_options.keys()))
        riwayat_preeklamsia = get_value(preeklamsia,preeklamsia_options)
        
    # Model ML
    
    # Kode untuk prediksi
    prediksi = ''
    
    # Membuat tombol untuk klasifikasi level risiko
    if st.button('Klasifikasi Level Risiko'):
        prediksi = preeklamsia_risk_level([tinggi_badan, berat_badan, tekanan_darah_sistolik, tekanan_darah_diastolik, usia, paritas, riwayat_hipertensi, riwayat_preeklamsia])
    
    st.success(prediksi)
    

# Halaman tentang web app    
if (selected == 'Tentang Sistem'):
    
    # Judul Halaman
    st.title('Tentang Sistem Deteksi Dini Risiko Preeklamsia') 

    
    
