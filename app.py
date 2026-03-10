import streamlit as st
import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# إعدادات الصفحة
st.set_page_config(page_title="Land Cover Classifier", layout="wide")

# الجزء المطلوب: Sidebar يحتوي على شرح مبسط
st.sidebar.title("حول النموذج 🤖")
st.sidebar.info("""
هذا التطبيق يستخدم نموذج **Decision Tree** تم تدريبه لتصنيف الصور الفضائية (Sentinel-2).
- **الأصناف المدعومة:**
  1. العمران (Urban)
  2. الزراعة (Agriculture)
  3. المياه (Water)
""")

st.title("🛰️ تطبيق تصنيف الصور الفضائية")

# الجزء المطلوب: Main Area - رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة القمر الصناعي (GeoTIFF)", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        # قراءة الصورة باستخدام rasterio
        with rasterio.open(uploaded_file) as src:
            img_data = src.read()
            meta = src.meta.copy()
            num_bands = src.count
            
            # عرض الصورة الأصلية (RGB افتراضي للعرض فقط)
            st.subheader("🖼️ الصورة الأصلية")
            # نأخذ أول 3 باندات للعرض السريع
            display_img = src.read([1, 2, 3])
            display_img = np.moveaxis(display_img, 0, -1)
            # عمل Normalize بسيط للعرض
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            st.image(display_img, caption="الصورة المرفوعة", use_container_width=True)

            # الجزء المطلوب: 3 قوائم منسدلة لاختيار الباندات
            st.write("---")
            st.subheader("⚙️ إعدادات التصنيف")
            col1, col2, col3 = st.columns(3)
            with col1:
                r_band = st.selectbox("باند الأحمر (Red)", range(1, num_bands + 1), index=0)
            with col2:
                g_band = st.selectbox("باند الأخضر (Green)", range(1, num_bands + 1), index=1)
            with col3:
                b_band = st.selectbox("باند الأزرق (Blue)", range(1, num_bands + 1), index=2)

            # الجزء المطلوب: زر بدء التصنيف
            if st.button("🚀 ابدأ التصنيف الآن"):
                with st.spinner('جاري معالجة الصورة...'):
                    # تحميل الموديل المحفوظ
                    model = joblib.load('model.pkl')
                    
                    # استخراج الباندات المختارة
                    input_data = src.read([r_band, g_band, b_band])
                    c, h, w = input_data.shape
                    
                    # تحويل البيانات لشكل جدول (pixels, 3)
                    flat_data = input_data.reshape(c, -1).T
                    
                    # تنفيذ التصنيف
                    prediction = model.predict(flat_data)
                    # تحويل النتائج إلى أرقام صحيحة صريحة (Integer) للتأكد من توافقها
                    prediction = prediction.astype(np.uint8)
                    # إعادة التشكيل للصورة الأصلية
                    classified_img = prediction.reshape(h, w)

                    # الجزء المطلوب: عرض الصورة المصنفة مع Legend
                    st.write("---")
                    st.subheader("✅ نتيجة التصنيف")
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    # استخدام ألوان واضحة: Urban (رمادي)، Agri (أخضر)، Water (أزرق)
                    # ملاحظة: تأكد أن القيم 1, 2, 3 تطابق كودك
                    im = ax.imshow(classified_img, cmap='terrain') 
                    plt.title("Classified Image")
                    plt.axis('off')
                    
                    # إضافة Legend
                    st.pyplot(fig)
                    st.info("الألوان تعبر عن الأصناف (مثال: أزرق = مياه، أخضر = زراعة، بني/رمادي = عمران)")

                    # الجزء المطلوب: تنزيل النتيجة بصيغة GeoTIFF
                    meta.update(count=1, dtype='uint8')
                    out_bytes = BytesIO()
                    # حفظ الملف في الذاكرة لتنزيله
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_img.astype('uint8'), 1)
                        data = memfile.read()
                    
                    st.download_button(
                        label="📥 تنزيل الصورة المصنفة كـ GeoTIFF",
                        data=data,
                        file_name="classified_result.tif",
                        mime="image/tiff"
                    )

    except Exception as e:
        st.error(f"❌ حدث خطأ: تأكد أن الملف بصيغة GeoTIFF صحيحة. التفاصيل: {e}")

else:

    st.warning("الرجاء رفع ملف الصورة للبدء.")
