import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات واجهة المهندس محمد حبوب
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")
st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.info("التصنيف باستخدام Decision Tree + MinMaxScaler")

st.title("🛰️ نظام تصنيف الغطاء الأرضي لقطاع غزة")

# 2. تحميل الموديل والسكيلر (تأكد أن الملفات مرفوعة بجانب الكود)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: تأكد من وجود model.pkl و scaler.pkl. التفاصيل: {e}")
        return None, None

model, scaler = load_assets()

uploaded_file = st.file_uploader("ارفع صورة غزة (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None and model is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            # قراءة أول 3 باندات (التي تدرب عليها الموديل)
            data = src.read([1, 2, 3]).astype(np.float64)
            c, h, w = data.shape
            
            # تحويل المصفوفة لشكل بكسلات (Flatten)
            flat_pixels = data.reshape(c, -1).T
            
            # --- الحل السحري: تحويل البيانات لـ DataFrame لضمان تطابق الأسماء ---
            # الموديل والسكيلر يبحثان عن SAMPLE_1, SAMPLE_2, SAMPLE_3
            df_pixels = pd.DataFrame(flat_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
            
            # 3. المعالجة والتنبؤ
            if st.button("🚀 تنفيذ التصنيف الدقيق"):
                with st.spinner('جاري معالجة البيانات وتطبيق السكيلر...'):
                    # تطبيق السكيلر المحفوظ من جوبيتر
                    scaled_pixels = scaler.transform(df_pixels)
                    
                    # إعادة تحويل البيانات لـ DataFrame بعد السكيلر للحفاظ على الأسماء للموديل
                    df_scaled = pd.DataFrame(scaled_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    # التنبؤ
                    prediction = model.predict(df_scaled)
                    
                    # تحويل المخرجات النصية (urban, water...) لأرقام للرسم
                    unique_labels = np.unique(prediction)
                    label_to_int = {label: i for i, label in enumerate(unique_labels)}
                    numeric_pred = np.array([label_to_int[p] for p in prediction])
                    
                    classified_img = numeric_pred.reshape(h, w)
                    
                    # 4. العرض والنتائج
                    st.write("---")
                    st.subheader("✅ خريطة التصنيف النهائية")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    # استخدام cmap='terrain' يعطي ألواناً منطقية (أخضر للزرع، أزرق للماء)
                    im = ax.imshow(classified_img, cmap='terrain')
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    # وسيلة الإيضاح (Legend)
                    st.write("**دليل الأصناف المكتشفة:**")
                    cols = st.columns(len(unique_labels))
                    for label, idx in label_to_int.items():
                        cols[idx].info(f"صنف {idx}: {label}")
                        
                    # زر تنزيل النتيجة (GeoTIFF)
                    meta = src.meta.copy()
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img.astype(np.uint8), 1)
                        st.download_button("📥 تحميل الخريطة المصنفة", memfile.read(), "Gaza_Classified.tif")

    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {e}")
