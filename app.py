import streamlit as st
import rasterio
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import LabelEncoder

# 1. إعدادات الصفحة
st.set_page_config(page_title="Land Cover Classifier", layout="wide")

# 2. القائمة الجانبية (Sidebar)
st.sidebar.title("مشروع تصنيف الغطاء الأرضي 🛰️")
st.sidebar.markdown("---")
st.sidebar.write("**إشراف:** المهندس محمد حبوب")
st.sidebar.info("""
هذا النظام يقوم بتصنيف صور Sentinel-2 إلى 3 فئات أساسية:
1. **Urban** (عمران)
2. **Agriculture** (زراعة)
3. **Water** (مياه)
""")

st.title("🛰️ نظام التصنيف الآلي للصور الفضائية")

# 3. رفع الملف
uploaded_file = st.file_uploader("ارفع صورة GeoTIFF", type=['tif', 'tiff'])

if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            meta = src.meta.copy()
            num_bands = src.count
            
            # عرض المعاينة الأصلية
            st.subheader("🖼️ معاينة الصورة الأصلية")
            if num_bands >= 3:
                # نأخذ الباندات 1,2,3 للعرض فقط ونقوم بعمل Normalization بسيط لتحسين الصورة
                preview = src.read([1, 2, 3])
                preview = np.moveaxis(preview, 0, -1)
                preview_scaled = (preview - preview.min()) / (preview.max() - preview.min() + 1e-5)
                st.image(preview_scaled, caption="الصورة الأصلية (RGB)", use_container_width=True)
            
            st.write("---")
            st.subheader("⚙️ إعدادات النموذج")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                r_idx = st.selectbox("باند الأحمر (Red)", range(1, num_bands + 1), index=0)
            with col2:
                g_idx = st.selectbox("باند الأخضر (Green)", range(1, num_bands + 1), index=1)
            with col3:
                b_idx = st.selectbox("باند الأزرق (Blue)", range(1, num_bands + 1), index=2)

            if st.button("🚀 تنفيذ التصنيف"):
                with st.spinner('جاري تحليل البكسلات...'):
                    # تحميل الموديل
                    model = joblib.load('model.pkl')
                    
                    # قراءة الباندات المختارة (القيم الأصلية ضرورية للنموذج)
                    input_bands = src.read([r_idx, g_idx, b_idx])
                    c, h, w = input_bands.shape
                    
                    # تحويل المصفوفة إلى شكل (Pixels, Features)
                    flat_pixels = input_bands.reshape(c, -1).T
                    
                    # عملية التنبؤ
                    raw_prediction = model.predict(flat_pixels)
                    
                    # التعامل مع المخرجات سواء كانت نصية أو رقمية
                    if np.issubdtype(raw_prediction.dtype, np.number):
                        final_pred = raw_prediction.astype(np.uint8)
                    else:
                        le = LabelEncoder()
                        final_pred = le.fit_transform(raw_prediction).astype(np.uint8)
                    
                    # إعادة تشكيل المصفوفة لأبعاد الصورة
                    classified_map = final_pred.reshape(h, w)

                    # 4. عرض النتائج والـ Legend
                    st.write("---")
                    st.subheader("✅ خريطة الغطاء الأرضي الناتجة")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # استخدام 'tab10' أو 'Set1' لإعطاء ألوان متباينة جداً للفئات
                    im = ax.imshow(classified_map, cmap='viridis') 
                    plt.colorbar(im, ax=ax, label="الفئات المصنفة")
                    plt.axis('off')
                    st.pyplot(fig)
                    
                    st.success("تم التصنيف! الألوان المختلفة تعبر عن تصنيفات الأرض (عمران، زراعة، ماء).")

                    # 5. زر التنزيل
                    meta.update(count=1, dtype='uint8')
                    with rasterio.MemoryFile() as memfile:
                        with memfile.open(**meta) as dataset:
                            dataset.write(classified_map, 1)
                        data = memfile.read()
                    
                    st.download_button(
                        label="📥 تحميل الخريطة المصنفة (GeoTIFF)",
                        data=data,
                        file_name="Land_Cover_Result.tif",
                        mime="image/tiff"
                    )

    except Exception as e:
        st.error(f"حدث خطأ أثناء المعالجة: {str(e)}")
else:
    st.info("يرجى رفع ملف الصورة للبدء.")
