import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. إعدادات الواجهة (طلب المهندس)
st.set_page_config(page_title="Gaza Satellite Classifier", layout="wide")

# Sidebar: شرح النموذج
st.sidebar.title("مشروع المهندس محمد حبوب 🛰️")
st.sidebar.info("""
**نظام تصنيف صور غزة:**
- الموديل: Decision Tree.
- المعالجة: MinMaxScaler (لحل مشكلة اللون الأزرق).
- الفئات: زراعة، عمران، مياه.
""")

st.title("🛰️ نظام تصنيف الغطاء الأرضي لقطاع غزة")

# 2. تحميل الموديل والسكيلر (تأكد من وجود الملفين في GitHub)
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ خطأ: تأكد من رفع model.pkl و scaler.pkl بجانب هذا الملف.")

# 3. رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة غزة (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None:
    with rasterio.open(uploaded_file) as src:
        # عرض المعاينة الأصلية
        st.subheader("🖼️ معاينة الصورة الأصلية")
        img_data = src.read([1, 2, 3])
        img_display = np.moveaxis(img_data, 0, -1).astype(float)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-5)
        st.image(img_display, use_container_width=True)

        # 4. قوائم اختيار الباندات (متطلبات المهندس)
        st.write("---")
        st.subheader("⚙️ إعدادات اختيار النطاقات")
        col1, col2, col3 = st.columns(3)
        with col1: r = st.selectbox("باند SAMPLE_1 (Red)", range(1, src.count + 1), index=0)
        with col2: g = st.selectbox("باند SAMPLE_2 (Green)", range(1, src.count + 1), index=1)
        with col3: b = st.selectbox("باند SAMPLE_3 (Blue)", range(1, src.count + 1), index=2)

        # 5. زر تنفيذ التصنيف
        if st.button("🚀 تنفيذ التصنيف النهائي"):
            with st.spinner('جاري معالجة البكسلات...'):
                # قراءة البيانات
                input_data = src.read([r, g, b]).astype(np.float64)
                c, h, w = input_data.shape
                
                # تحويل البيانات لجدول بأسماء الأعمدة الصحيحة (لحل خطأ Mismatch)
                flat_pixels = input_data.reshape(c, -1).T
                df_input = pd.DataFrame(flat_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                
                # --- الخطوة السحرية: تحويل القيم لـ (0-1) باستخدام السكيلر ---
                scaled_pixels = scaler.transform(df_input)
                # إعادة تسمية الأعمدة بعد السكيلر ليرضاها الموديل
                df_scaled = pd.DataFrame(scaled_pixels, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                
                # التنبؤ
                prediction = model.predict(df_scaled)
                
                # تحويل النتائج لأرقام للرسم
                unique_labels = np.unique(prediction)
                label_to_int = {label: i for i, label in enumerate(unique_labels)}
                numeric_pred = np.array([label_to_int[p] for p in prediction])
                
                classified_img = numeric_pred.reshape(h, w).astype(np.uint8)

                # 6. عرض النتيجة النهائية والـ Legend
                st.subheader("✅ نتيجة التصنيف الدقيقة")
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(classified_img, cmap='terrain') # تدرج لوني طبيعي
                plt.axis('off')
                st.pyplot(fig)
                
                # وسيلة الإيضاح (Legend)
                st.write("**دليل الأصناف:**")
                leg_cols = st.columns(len(unique_labels))
                for label, idx in label_to_int.items():
                    leg_cols[idx].success(f"{label}")

                # 7. زر التنزيل (Download)
                meta = src.meta.copy()
                meta.update(count=1, dtype='uint8')
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(**meta) as dataset:
                        dataset.write(classified_img, 1)
                    st.download_button("📥 تحميل الخريطة المصنفة", memfile.read(), "Gaza_Classified.tif")
