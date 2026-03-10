import streamlit as st
import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd  # أضفنا بانداز لضمان توافق أسماء الأعمدة

st.set_page_config(page_title="Land Cover Classifier", layout="wide")

st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.info("تصنيف Sentinel-2: عمران، زراعة، مياه")

st.title("🛰️ نظام التصنيف الاحترافي للصور الفضائية")

uploaded_file = st.file_uploader("ارفع صورة GeoTIFF", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            num_bands = src.count
            meta = src.meta.copy()
            
            # عرض الصورة الأصلية
            st.subheader("🖼️ معاينة الصورة")
            if num_bands >= 3:
                rgb = src.read([1, 2, 3])
                rgb_norm = np.moveaxis(rgb, 0, -1)
                # تحسين العرض (Scaling for display only)
                rgb_norm = (rgb_norm - rgb_norm.min()) / (rgb_norm.max() - rgb_norm.min() + 1e-5)
                st.image(rgb_norm, use_container_width=True)

            # اختيار الباندات
            st.write("---")
            col1, col2, col3 = st.columns(3)
            with col1: r = st.selectbox("Red Band", range(1, num_bands + 1), index=0)
            with col2: g = st.selectbox("Green Band", range(1, num_bands + 1), index=1)
            with col3: b = st.selectbox("Blue Band", range(1, num_bands + 1), index=2)

            if st.button("🚀 تنفيذ التصنيف النهائي"):
                with st.spinner('جاري التحليل...'):
                    model = joblib.load('model.pkl')
                    
                    # قراءة البيانات الخام (Raw Data) كما هي في ملف الـ CSV
                    input_data = src.read([r, g, b])
                    c, h, w = input_data.shape
                    flat_pixels = input_data.reshape(c, -1).T
                    
                    # تحويل البيانات لـ DataFrame بنفس أسماء الأعمدة التي تدرب عليها الموديل
                    # ملاحظة: تأكد أن B1, B2, B3 هي نفس الأسماء في ملفك الـ CSV
                    df_pixels = pd.DataFrame(flat_pixels, columns=['B1', 'B2', 'B3'])
                    
                    # التنبؤ
                    prediction = model.predict(df_pixels)
                    
                    # تحويل النتائج لأرقام إذا كانت نصوصاً
                    if not np.issubdtype(prediction.dtype, np.number):
                        from sklearn.preprocessing import LabelEncoder
                        prediction = LabelEncoder().fit_transform(prediction)
                    
                    classified_img = prediction.reshape(h, w).astype(np.uint8)

                    # العرض مع Legend واضح
                    st.write("---")
                    st.subheader("✅ خريطة الغطاء الأرضي")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    # استخدام خريطة ألوان متباينة جداً (Set1 أو tab10)
                    im = ax.imshow(classified_img, cmap='tab10') 
                    cbar = plt.colorbar(im, ax=ax, ticks=np.unique(classified_img))
                    cbar.set_label('Classes (0, 1, 2...)')
                    plt.axis('off')
                    st.pyplot(fig)

                    # زر التنزيل
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        st.download_button("📥 تحميل النتيجة", memfile.read(), "result.tif", "image/tiff")

    except Exception as e:
        st.error(f"خطأ: {e}")
