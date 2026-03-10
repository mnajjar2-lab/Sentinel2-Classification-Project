import streamlit as st
import rasterio
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# 1. إعدادات الصفحة (Streamlit Config)
st.set_page_config(page_title="Land Cover Classifier", layout="wide")

# 2. Sidebar: شرح مبسط للنموذج (طلب المهندس)
st.sidebar.title("🤖 حول النموذج المستخدم")
st.sidebar.info("""
هذا التطبيق مخصص لتصنيف صور Sentinel-2 باستخدام خوارزمية **Decision Tree**.
- **الهدف:** تصنيف الغطاء الأرضي إلى (عمران، زراعة، مياه).
- **ملاحظة فنية:** تم ضبط الكود ليعالج تباين نطاقات البيانات (Scaling) لضمان دقة النتائج.
""")

st.title("🛰️ نظام التصنيف الاحترافي للصور الفضائية")

# 3. Main Area: رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة القمر الصناعي (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            num_bands = src.count
            
            # عرض الصورة الأصلية (متطلب المهندس)
            st.subheader("🖼️ معاينة الصورة الأصلية")
            if num_bands >= 3:
                # قراءة أول 3 باندات للعرض فقط
                preview_raw = src.read([1, 2, 3])
                preview_norm = np.moveaxis(preview_raw, 0, -1)
                # عمل Normalize للعرض فقط
                preview_norm = (preview_norm - preview_norm.min()) / (preview_norm.max() - preview_norm.min() + 1e-5)
                st.image(preview_norm, caption="الصورة المرفوعة (RGB)", use_container_width=True)

            # 4. اختيار الباندات (متطلب المهندس - قوائم منسدلة)
            st.write("---")
            st.subheader("⚙️ إعدادات التصنيف")
            st.write("اختر الباندات التي تطابق ترتيب التدريب (Sample 1, 2, 3):")
            col1, col2, col3 = st.columns(3)
            with col1:
                r_idx = st.selectbox("Red Band (SAMPLE_1)", range(1, num_bands + 1), index=0)
            with col2:
                g_idx = st.selectbox("Green Band (SAMPLE_2)", range(1, num_bands + 1), index=1 if num_bands > 1 else 0)
            with col3:
                b_idx = st.selectbox("Blue Band (SAMPLE_3)", range(1, num_bands + 1), index=2 if num_bands > 2 else 0)

            # 5. زر بدء التصنيف
            if st.button("🚀 ابدأ عملية التصنيف"):
                with st.spinner('جاري معالجة البيانات وتطبيق النموذج...'):
                    # تحميل الموديل
                    model = joblib.load('model.pkl')
                    
                    # قراءة الباندات المختارة
                    input_data = src.read([r_idx, g_idx, b_idx]).astype(float)
                    c, h, w = input_data.shape
                    
                    # --- حل مشكلة الـ Scaling (القيم العشرية) ---
                    for i in range(c):
                        b_min, b_max = input_data[i].min(), input_data[i].max()
                        input_data[i] = (input_data[i] - b_min) / (b_max - b_min + 1e-5)
                    
                    # تحويل البيانات لشكل جدول
                    flat_data = input_data.reshape(c, -1).T
                    
                    # --- حل مشكلة الـ Feature Names (الخطأ الذي ظهر لك) ---
                    # قمنا بتغيير الأسماء لتطابق 'SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3' كما رآها الموديل عند التدريب
                    df_input = pd.DataFrame(flat_data, columns=['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3'])
                    
                    # التنبؤ (Prediction)
                    prediction = model.predict(df_input)
                    
                    # تحويل المخرجات النصية (مثل 'urban') إلى أرقام للرسم
                    if not np.issubdtype(prediction.dtype, np.number):
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        prediction = le.fit_transform(prediction)
                        classes_list = le.classes_
                    else:
                        classes_list = np.unique(prediction)

                    # إعادة التشكيل للصورة الأصلية
                    classified_img = prediction.reshape(h, w).astype(np.uint8)

                    # 6. عرض الصورة المصنفة مع Legend (متطلب المهندس)
                    st.write("---")
                    st.subheader("✅ نتيجة التصنيف النهائية")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    # استخدام خريطة ألوان متباينة
                    im = ax.imshow(classified_img, cmap='terrain') 
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    # إظهار وسيلة إيضاح (Legend)
                    st.write("**دليل الأصناف (Legend):**")
                    legend_cols = st.columns(len(classes_list))
                    for idx, label in enumerate(classes_list):
                        legend_cols[idx].info(f"اللون {idx}: {label}")

                    # 7. تنزيل النتيجة بصيغة GeoTIFF (متطلب المهندس)
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        data = memfile.read()
                    
                    st.download_button(
                        label="📥 تنزيل الخريطة المصنفة كـ GeoTIFF",
                        data=data,
                        file_name="Gaza_Classification_Result.tif",
                        mime="image/tiff"
                    )

    except Exception as e:
        # رسالة خطأ واضحة (متطلب المهندس)
        st.error(f"❌ خطأ: لم نتمكن من معالجة الملف. التفاصيل: {e}")

else:
    st.warning("الرجاء رفع ملف GeoTIFF للبدء.")
