import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. إعدادات الواجهة الاحترافية (طلبات المهندس)
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")

# Sidebar: شرح النموذج
st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.info("""
**نظام تصنيف صور غزة:**
- النموذج: Decision Tree (Max Depth: 10).
- المدخلات: Sentinel-2 Bands.
- المعالجة: يتم عمل Normalization تلقائي لمطابقة بيانات التدريب (0-1).
""")

st.title("🛰️ تطبيق تصنيف الصور الفضائية - الحل النهائي")

uploaded_file = st.file_uploader("ارفع صورة GeoTIFF غزة", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            
            # عرض المعاينة الأصلية RGB
            st.subheader("🖼️ معاينة الصورة الأصلية")
            rgb = src.read([1, 2, 3])
            rgb_view = np.moveaxis(rgb, 0, -1).astype(float)
            # Normalize للعرض فقط
            rgb_view = (rgb_view - rgb_view.min()) / (rgb_view.max() - rgb_view.min() + 1e-5)
            st.image(rgb_view, use_container_width=True)

            # إعدادات التصنيف (قوائم منسدلة)
            st.write("---")
            st.subheader("⚙️ إعدادات مطابقة البيانات")
            col1, col2, col3 = st.columns(3)
            with col1: r = st.selectbox("باند SAMPLE_1 (Red)", range(1, src.count + 1), index=0)
            with col2: g = st.selectbox("باند SAMPLE_2 (Green)", range(1, src.count + 1), index=1)
            with col3: b = st.selectbox("باند SAMPLE_3 (Blue)", range(1, src.count + 1), index=2)

            if st.button("🚀 تنفيذ التصنيف الآن"):
                with st.spinner('جاري تحليل بكسلات الصورة...'):
                    # تحميل الموديل
                    model = joblib.load('model.pkl')
                    
                    # قراءة البيانات وإجبارها على نوع Float64
                    input_data = src.read([r, g, b]).astype(np.float64)
                    c, h, w = input_data.shape
                    
                    # --- الحل السحري لمشكلة اللون الأزرق (Scaling) ---
                    # تحويل قيم الصورة لتصبح بين (0 و1) لتطابق ملف الـ CSV تماماً
                    for i in range(c):
                        b_min = input_data[i].min()
                        b_max = input_data[i].max()
                        if b_max > b_min:
                            input_data[i] = (input_data[i] - b_min) / (b_max - b_min)
                        else:
                            input_data[i] = 0.0
                    
                    # تحويل لجدول بأسماء SAMPLE_1, 2, 3
                    flat_pixels = input_data.reshape(c, -1).T
                    df_input = pd.DataFrame(flat_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    # التنبؤ (Prediction)
                    prediction = model.predict(df_input)
                    
                    # تحويل المخرجات النصية (مثل 'urban') لأرقام
                    le = LabelEncoder()
                    pred_numeric = le.fit_transform(prediction.astype(str))
                    classes_labels = le.classes_
                    
                    classified_img = pred_numeric.reshape(h, w).astype(np.uint8)

                    # عرض النتيجة والـ Legend
                    st.write("---")
                    st.subheader("✅ نتيجة التصنيف النهائية")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    # استخدام خريطة ألوان متباينة
                    im = ax.imshow(classified_img, cmap='terrain') 
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    # الـ Legend (وسيلة الإيضاح)
                    st.write("**دليل الأصناف المكتشفة:**")
                    leg_cols = st.columns(len(classes_labels))
                    for idx, label in enumerate(classes_labels):
                        leg_cols[idx].success(f"اللون {idx}: {label}")

                    # زر التنزيل GeoTIFF
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        st.download_button("📥 تنزيل نتيجة غزة المصنفة", memfile.read(), "Gaza_Classification.tif")

    except Exception as e:
        st.error(f"⚠️ حدث خطأ: تأكد من ملف الموديل والبيانات. التفاصيل: {e}")
