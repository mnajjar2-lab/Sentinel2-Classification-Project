import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. إعدادات واجهة المهندس محمد حبوب
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")
st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.info("النموذج: Decision Tree (Max Depth: 10)") # كما في كود التدريب

st.title("🛰️ نظام تصنيف الصور الفضائية - النسخة النهائية")

uploaded_file = st.file_uploader("ارفع صورة GeoTIFF الخاصة بغزة", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            
            # معاينة الصورة RGB
            st.subheader("🖼️ الصورة الأصلية المرفوعة")
            rgb = src.read([1, 2, 3])
            rgb_norm = np.moveaxis(rgb, 0, -1)
            rgb_norm = (rgb_norm - rgb_norm.min()) / (rgb_norm.max() - rgb_norm.min() + 1e-5)
            st.image(rgb_norm, use_container_width=True)

            # اختيار الباندات (القوائم المنسدلة)
            st.write("---")
            st.subheader("⚙️ مطابقة الباندات مع التدريب")
            col1, col2, col3 = st.columns(3)
            with col1: r = st.selectbox("باند B1 (الأحمر)", range(1, src.count + 1), index=0)
            with col2: g = st.selectbox("باند B2 (الأخضر)", range(1, src.count + 1), index=1)
            with col3: b = st.selectbox("باند B3 (الأزرق)", range(1, src.count + 1), index=2)

            if st.button("🚀 ابدأ تصنيف بكسلات غزة"):
                with st.spinner('جاري التحليل...'):
                    # تحميل الموديل
                    model = joblib.load('model.pkl')
                    
                    # قراءة البيانات (يجب أن تكون بنفس طبيعة بيانات التدريب)
                    input_data = src.read([r, g, b]).astype(np.float64)
                    c, h, w = input_data.shape
                    
                    # ملاحظة: إذا كان ملف CSV التدريب يحتوي على قيم (0-1)، اترك الـ Scaling. 
                    # إذا كان يحتوي على أرقام كبيرة، امسح الـ Scaling التالي:
                    # لضمان الدقة، سنفترض هنا أن التدريب كان على قيم خام (Raw Values)
                    
                    flat_pixels = input_data.reshape(c, -1).T
                    
                    # التعديل الجوهري: مطابقة أسماء الأعمدة مع كود التدريب
                    df_pixels = pd.DataFrame(flat_pixels, columns=['B1', 'B2', 'B3'])
                    
                    # تنفيذ التنبؤ
                    prediction = model.predict(df_pixels)
                    
                    # تحويل المخرجات (في حال كانت نصوصاً أو أرقاماً)
                    le = LabelEncoder()
                    pred_numeric = le.fit_transform(prediction.astype(str))
                    classes_found = le.classes_
                    
                    classified_img = pred_numeric.reshape(h, w).astype(np.uint8)

                    # عرض النتيجة ووسيلة الإيضاح (Legend)
                    st.write("---")
                    st.subheader("✅ خريطة الغطاء الأرضي الناتجة")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(classified_img, cmap='terrain') 
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    # وسيلة الإيضاح
                    st.write("**دليل الأصناف (Legend):**")
                    legend_cols = st.columns(len(classes_found))
                    for i, cls in enumerate(classes_found):
                        # ربط الأسماء (Urban, Agri, Water) بالأرقام
                        legend_cols[i].success(f"فئة {i}: {cls}")

                    # تنزيل النتيجة GeoTIFF
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        st.download_button("📥 تحميل الخريطة النهائية", memfile.read(), "Gaza_Classified.tif")

    except Exception as e:
        st.error(f"حدث خطأ فني: {e}")
