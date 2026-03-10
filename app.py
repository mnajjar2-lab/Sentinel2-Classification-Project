import streamlit as st
import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

# 1. إعدادات الصفحة
st.set_page_config(page_title="Land Cover Classifier", layout="wide")

# 2. الجزء المطلوب: Sidebar يحتوي على شرح مبسط للمشروع
st.sidebar.title("مشروع محمد - تصنيف الغطاء الأرضي 🛰️")
st.sidebar.info("""
هذا التطبيق يستخدم نموذج **Decision Tree** تم تدريبه لتصنيف الصور الفضائية (Sentinel-2).
- **الأصناف المدعومة:**
  1. العمران (Urban)
  2. الزراعة (Agriculture)
  3. المياه (Water)
""")

st.title("🛰️ تطبيق تصنيف الصور الفضائية الذكي")

# 3. الجزء المطلوب: Main Area - رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة القمر الصناعي بصيغة (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        # قراءة الصورة باستخدام rasterio
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            num_bands = src.count
            
            # عرض الصورة الأصلية (RGB)
            st.subheader("🖼️ الصورة الأصلية")
            # قراءة أول 3 باندات للعرض فقط
            if num_bands >= 3:
                display_img = src.read([1, 2, 3])
                display_img = np.moveaxis(display_img, 0, -1)
                # عمل Normalize لضمان ظهور الألوان بشكل صحيح
                display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
                st.image(display_img, caption="معاينة الصورة المرفوعة", use_container_width=True)
            else:
                st.warning("الصورة تحتوي على أقل من 3 باندات للعرض الملون.")

            # 4. الجزء المطلوب: اختيار الباندات من قوائم منسدلة
            st.write("---")
            st.subheader("⚙️ إعدادات التصنيف")
            st.write("اختر الباندات التي تم استخدامها أثناء تدريب النموذج (الترتيب مهم):")
            col1, col2, col3 = st.columns(3)
            with col1:
                r_band = st.selectbox("باند الأحمر (Red)", range(1, num_bands + 1), index=0)
            with col2:
                g_band = st.selectbox("باند الأخضر (Green)", range(1, num_bands + 1), index=1)
            with col3:
                b_band = st.selectbox("باند الأزرق (Blue)", range(1, num_bands + 1), index=2)

            # 5. الجزء المطلوب: تنفيذ التصنيف عند الضغط على الزر
            if st.button("🚀 ابدأ عملية التصنيف"):
                with st.spinner('جاري قراءة البكسلات وتطبيق النموذج...'):
                    # تحميل الموديل المحفوظ
                    model = joblib.load('model.pkl')
                    
                    # استخراج الباندات المختارة وتحضيرها
                    input_data = src.read([r_band, g_band, b_band])
                    c, h, w = input_data.shape
                    
                    # تحويل البيانات لشكل جدول (Pixels x Features)
                    flat_data = input_data.reshape(c, -1).T
                    
                    # التنبؤ (Prediction)
                    prediction = model.predict(flat_data)
                    
                    # --- معالجة احترافية لأنواع البيانات (تجنب خطأ dtype object) ---
                    if np.issubdtype(prediction.dtype, np.number):
                        # إذا كانت النتائج أرقاماً (1, 2, 3)
                        final_prediction = prediction.astype(np.uint8)
                    else:
                        # إذا كانت النتائج نصوصاً ('urban', 'water')
                        le = LabelEncoder()
                        final_prediction = le.fit_transform(prediction).astype(np.uint8)
                    
                    # إعادة التشكيل لأبعاد الصورة الأصلية
                    classified_img = final_prediction.reshape(h, w)

                    # 6. الجزء المطلوب: عرض النتيجة مع Legend
                    st.write("---")
                    st.subheader("✅ نتيجة التصنيف النهائية")
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    # استخدام خريطة ألوان terrain المناسبة لغطاء الأرض
                    im = ax.imshow(classified_img, cmap='terrain') 
                    plt.colorbar(im, ax=ax, ticks=np.unique(classified_img), label="فئات الغطاء الأرضي")
                    plt.title("Classification Result")
                    plt.axis('off')
                    
                    st.pyplot(fig)
                    st.success("تم التصنيف بنجاح! يمكنك الآن رؤية توزيع (المياه، الزراعة، والعمران).")

                    # 7. الجزء المطلوب: تنزيل النتيجة بصيغة GeoTIFF
                    meta.update(count=1, dtype='uint8')
                    
                    # حفظ الملف في الذاكرة (MemoryFile) لتوفير زر تحميل سريع
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img, 1)
                        data = memfile.read()
                    
                    st.download_button(
                        label="📥 تنزيل الخريطة المصنفة (GeoTIFF)",
                        data=data,
                        file_name="classified_land_cover.tif",
                        mime="image/tiff"
                    )

    except Exception as e:
        st.error(f"❌ خطأ تقني: {e}")
        st.info("تأكد من أن الموديل (model.pkl) متوافق مع عدد الباندات المختار.")

else:
    st.warning("ننتظر منك رفع ملف الصورة (GeoTIFF) للبدء في التحليل.")
