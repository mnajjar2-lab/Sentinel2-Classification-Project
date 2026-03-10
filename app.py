import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الواجهة (طلب المهندس محمد حبوب)
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")

# Sidebar: شرح النموذج
st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.markdown("""
### وصف المشروع
نظام تصنيف صور غزة باستخدام **Decision Tree**.
- **المعالجة:** MinMaxScaler.
- **الأصناف:** Urban, Agricultureal, Water.
""")

st.title("🛰️ تطبيق تصنيف الصور الفضائية النهائي")

# 2. تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ تأكد من رفع model.pkl و scaler.pkl: {e}")
        return None, None

model, scaler = load_assets()

# 3. رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة غزة (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None and model is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            # عرض المعاينة الأصلية
            st.subheader("🖼️ الصورة الأصلية")
            img_data = src.read([1, 2, 3])
            img_display = np.moveaxis(img_data, 0, -1).astype(float)
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-5)
            st.image(img_display, use_container_width=True)

            # 4. قوائم اختيار الباندات (متطلبات المهندس)
            st.write("---")
            st.subheader("⚙️ إعدادات الباندات")
            col1, col2, col3 = st.columns(3)
            with col1: b1 = st.selectbox("باند SAMPLE_1", range(1, src.count + 1), index=0)
            with col2: b2 = st.selectbox("باند SAMPLE_2", range(1, src.count + 1), index=1)
            with col3: b3 = st.selectbox("باند SAMPLE_3", range(1, src.count + 1), index=2)

            if st.button("🚀 تنفيذ التصنيف"):
                with st.spinner('جاري التحليل...'):
                    # قراءة البيانات المختارة
                    data = src.read([b1, b2, b3]).astype(np.float64)
                    c, h, w = data.shape
                    
                    # تحويل المصفوفة إلى DataFrame بالأسماء الصحيحة (لحل مشكلة الخطأ)
                    flat_pixels = data.reshape(c, -1).T
                    df_input = pd.DataFrame(flat_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    # --- الخطوة الحاسمة ---
                    # تطبيق السكيلر ثم التنبؤ باستخدام DataFrame
                    scaled_pixels = scaler.transform(df_input)
                    df_scaled = pd.DataFrame(scaled_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    prediction = model.predict(df_scaled)
                    
                    # تحويل المخرجات النصية لأرقام للرسم
                    unique_labels = np.unique(prediction)
                    label_map = {label: i for i, label in enumerate(unique_labels)}
                    numeric_pred = np.array([label_map[p] for p in prediction])
                    
                    classified_img = numeric_pred.reshape(h, w).astype(np.uint8)

                    # 5. عرض النتيجة والـ Legend
                    st.subheader("✅ خريطة الغطاء الأرضي المصنفة")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(classified_img, cmap='terrain')
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    st.write("**دليل الأصناف:**")
                    legend_cols = st.columns(len(unique_labels))
                    for label, idx in label_map.items():
                        legend_cols[idx].success(f"صنف {idx}: {label}")

                    # 6. تنزيل النتيجة
                    meta = src.meta.copy()
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        st.download_button("📥 تحميل النتيجة GeoTIFF", memfile.read(), "Gaza_Result.tif")

    except Exception as e:
        st.error(f"❌ خطأ في المعالجة: {e}")
