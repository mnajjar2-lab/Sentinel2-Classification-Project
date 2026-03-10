#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[3]:


# تحميل ملفك اللي استخرجته من QGIS
df = pd.read_csv('training_data.csv')

# التأكد من أسماء الأعمدة (SAMPLE_1, 2, 3) والفئات (info)
print(df.head())
print(df['info'].unique()) # لازم يطلع لك: urban, water, agricultureal


# In[4]:


# 1. تحديد الميزات والهدف
X = df[['SAMPLE_1', 'SAMPLE_2', 'SAMPLE_3']]
y = df['info']

# 2. إنشاء الـ Scaler وتحويل البيانات (Min-Max Scaling)
# هاد اللي بيخلي القيم بين 0 و 1 عشان الموديل يفهمها صح
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. تقسيم البيانات (70% تدريب، 30% تحقق)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[5]:


# تدريب الشجرة مع تحديد العمق 10 كما طلب المهندس
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

print("تم تدريب الموديل بنجاح!")


# In[6]:


y_pred = model.predict(X_val)

# 1. Accuracy
print(f"Accuracy Score: {accuracy_score(y_val, y_pred):.4f}")

# 2. Classification Report (Precision, Recall, F1)
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# 3. Confusion Matrix الرسم
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Gaza Project')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[7]:


# حفظ الموديل
joblib.dump(model, 'model.pkl')

# حفظ السكيلر (ضروري جداً عشان تطبيق الويب يشتغل صح)
joblib.dump(scaler, 'scaler.pkl')

print("مبروك! صار عندك ملفين: model.pkl و scaler.pkl جاهزين للرفع على GitHub")


# In[ ]:




