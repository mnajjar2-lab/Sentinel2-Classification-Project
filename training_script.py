import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. تحميل البيانات المستخرجة من QGIS
# تأكد أن الملف موجود في نفس المجلد
try:
    df = pd.read_csv('training_data.csv')
    print("تم تحميل البيانات بنجاح.")
except FileNotFoundError:
    print("خطأ: لم يتم العثور على ملف training_data.csv")

# 2. تحديد الميزات والهدف
# افترضنا أن الأعمدة هي B1, B2, B3 والتصنيف هو Class_ID
X = df[['B1', 'B2', 'B3']] 
y = df['Class_ID']

# 3. تقسيم البيانات (70% تدريب، 30% تحقق)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. بناء وتدريب نموذج شجرة القرار (Decision Tree)
# حددنا max_depth=10 كما هو مطلوب لتجنب التعقيد الزائد
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 5. تقييم النموذج وحساب المقاييس
y_pred = model.predict(X_val)
print(f"الدقة (Accuracy): {accuracy_score(y_val, y_pred):.4f}")
print("\nتقرير التصنيف المفصل:")
print(classification_report(y_val, y_pred))

# 6. رسم وحفظ مصفوفة الارتباك (Confusion Matrix)
cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Urban', 'Agri', 'Water'], 
            yticklabels=['Urban', 'Agri', 'Water'], ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
fig.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
print("تم حفظ مصفوفة الارتباك بنجاح.")

# 7. حفظ النموذج النهائي
joblib.dump(model, 'model.pkl')
print("تم حفظ النموذج باسم model.pkl")