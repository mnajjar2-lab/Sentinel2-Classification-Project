import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# إعدادات الصفحة
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")

st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.info("تصنيف Sentinel-2: عمران، زراعة، مياه")

st.title("🛰️ تطبيق تصنيف الصور الفضائية")

uploaded_file = st.file_uploader("ارفع صورة GeoTIFF", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            num_bands = src.count
            
            # عرض المعاينة
            st.subheader("🖼️ الصورة الأصلية")
            if num_bands >= 3:
                rgb = src.read([1, 2, 3])
                rgb_norm = np.moveaxis(rgb, 0, -1)
                rgb_norm = (rgb_norm - rgb_norm.min()) / (rgb_norm.max() - rgb_norm.min() + 1e-5)
                st.image(rgb_norm, use_container_width=True)

            # اختيار الباندات
            st.write("---")
            st.subheader("⚙️ إعدادات التصنيف")
            col1, col2, col3 = st.columns(3)
            with col1: r = st.selectbox("باند الأحمر (SAMPLE_1)", range(1, num_bands + 1), index=0)
            with col2: g = st.selectbox("باند الأخضر (SAMPLE_2)", range(1, num_bands + 1), index=1 if num_bands > 1 else 0)
            with col3: b = st.selectbox("باند الأزرق (SAMPLE_3)", range(1, num_bands + 1), index=2 if num_bands > 2 else 0)

            if st.button("🚀 تنفيذ التصنيف الآن"):
                with st.spinner('جاري التحليل...'):
                    # 1. تحميل الموديل
                    model = joblib.load('model.pkl')
                    
                    # 2. قراءة البيانات وتحويلها لـ float64 لتجنب أخطاء الـ dtype
                    input_data = src.read([r, g, b]).astype(np.float64)
                    c, h, w = input_data.shape
                    
                    # 3. الـ Scaling (تحويل القيم لنطاق 0-1 كما في الـ CSV)
                    for i in range(c):
                        b_min, b_max = input_data[i].min(), input_data[i].max()
                        input_data[i] = (input_data[i] - b_min) / (b_max - b_min + 1e-5)
                    
                    # 4. تحويل لـ DataFrame بالأسماء التي رآها الموديل (SAMPLE_1, 2, 3)
                    flat_pixels = input_data.reshape(c, -1).T
                    df_pixels = pd.DataFrame(flat_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    # 5. التنبؤ ومعالجة النتائج النصية (مثل 'urban')
                    prediction = model.predict(df_pixels)
                    
                    # تحويل المخرجات لأرقام للرسم
                    le = LabelEncoder()
                    # fit_transform تتعامل مع المخرجات سواء كانت نصوصاً أو أرقاماً
                    pred_numeric = le.fit_transform(prediction.astype(str))
                    classes_found = le.classes_
                    
                    classified_img = pred_numeric.reshape(h, w).astype(np.uint8)

                    # 6. العرض والـ Legend
                    st.write("---")
                    st.subheader("✅ نتيجة التصنيف")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(classified_img, cmap='terrain')
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    # عرض الـ Legend
                    st.write("**دليل الأصناف:**")
                    legend_cols = st.columns(len(classes_found))
                    for i, cls in enumerate(classes_found):
                        legend_cols[i].success(f"صنف {i}: {cls}")

                    # 7. التنزيل
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        st.download_button("📥 تحميل النتيجة", memfile.read(), "Gaza_Result.tif", "image/tiff")

    except Exception as e:
        st.error(f"حدث خطأ: {e}")
