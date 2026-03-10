import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# واجهة المهندس محمد حبوب
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")
st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")

st.title("🛰️ تصنيف الغطاء الأرضي باستخدام MinMaxScaler")

# تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

uploaded_file = st.file_uploader("ارفع صورة غزة (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        # قراءة الباندات وتحضير البيانات
        data = src.read([1, 2, 3]).astype(np.float64)
        c, h, w = data.shape
        
        # تحويل المصفوفة لشكل جدول (Flatten)
        flat_pixels = data.reshape(c, -1).T
        
        # تطبيق السكيلر (المسطرة) المحفوظة من التدريب
        scaled_pixels = scaler.transform(flat_pixels)
        
        # التنبؤ
        if st.button("🚀 تنفيذ التصنيف الدقيق"):
            prediction = model.predict(scaled_pixels)
            
            # تحويل المخرجات النصية لأرقام للرسم
            label_map = {label: idx for idx, label in enumerate(np.unique(prediction))}
            numeric_pred = np.array([label_map[p] for p in prediction])
            
            classified_img = numeric_pred.reshape(h, w)
            
            # العرض
            fig, ax = plt.subplots()
            ax.imshow(classified_img, cmap='terrain')
            plt.axis('off')
            st.pyplot(fig)
            
            # Legend
            st.write("**الأصناف المكتشفة:**")
            for label, idx in label_map.items():
                st.success(f"اللون {idx}: {label}")
