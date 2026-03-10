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
st.sidebar.info("تصنيف Sentinel-2 باستخدام Decision Tree")

st.title("🛰️ تطبيق تصنيف الصور الفضائية الاحترافي")

uploaded_file = st.file_uploader("ارفع صورة GeoTIFF غزة", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            
            # عرض المعاينة الأصلية
            st.subheader("🖼️ معاينة الصورة الأصلية")
            rgb = src.read([1, 2, 3])
            # تحويل القيم لـ Normalize للعرض فقط
            rgb_view = np.moveaxis(rgb, 0, -1)
            rgb_view = (rgb_view - rgb_view.min()) / (rgb_view.max() - rgb_view.min() + 1e-5)
            st.image(rgb_view, use_container_width=True)

            # إعدادات التصنيف
            st.write("---")
            st.subheader("⚙️ إعدادات مطابقة البيانات")
            col1, col2, col3 = st.columns(3)
            with col1: r = st.selectbox("باند SAMPLE_1 (Red)", range(1, src.count + 1), index=0)
            with col2: g = st.selectbox("باند SAMPLE_2 (Green)", range(1, src.count + 1), index=1)
            with col3: b = st.selectbox("باند SAMPLE_3 (Blue)", range(1, src.count + 1), index=2)

            if st.button("🚀 تنفيذ التصنيف النهائي"):
                with st.spinner('جاري تحليل بكسلات الصورة...'):
                    # 1. تحميل الموديل
                    model = joblib.load('model.pkl')
                    
                    # 2. قراءة البيانات وإجبارها على نوع Float64 (حل مشكلة dtype object)
                    input_data = src.read([r, g, b]).astype(np.float64)
                    c, h, w = input_data.shape
                    
                    # 3. تحويل البيانات لجدول بأسماء الأعمدة التي يطلبها الموديل (حل خطأ الأسماء)
                    flat_pixels = input_data.reshape(c, -1).T
                    df_input = pd.DataFrame(flat_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    # 4. التنبؤ ومعالجة النتائج النصية (حل خطأ 'urban')
                    prediction = model.predict(df_input)
                    
                    # تحويل المخرجات (سواء كانت نصوص أو أرقام) إلى أرقام للرسم
                    le = LabelEncoder()
                    pred_numeric = le.fit_transform(prediction.astype(str))
                    classes_labels = le.classes_
                    
                    classified_img = pred_numeric.reshape(h, w).astype(np.uint8)

                    # 5. عرض النتيجة النهائية مع Legend
                    st.write("---")
                    st.subheader("✅ نتيجة التصنيف")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(classified_img, cmap='terrain')
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    # إظهار وسيلة الإيضاح (Legend)
                    st.write("**دليل التصنيف (Legend):**")
                    leg_cols = st.columns(len(classes_labels))
                    for idx, label in enumerate(classes_labels):
                        leg_cols[idx].success(f"قيمة {idx}: {label}")

                    # 6. زر التحميل
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        st.download_button("📥 تحميل نتيجة غزة (TIF)", memfile.read(), "Gaza_Result.tif")

    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء المعالجة: {e}")
