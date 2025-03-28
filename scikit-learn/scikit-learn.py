# 1. 라이브러리 불러오기
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
# 2. 데이터 불러오기
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 3. EDA 시각화
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Survived')
plt.title("생존자/사망자 수")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("객실 등급별 생존자 수")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title("나이 분포별 생존율")
plt.show()

# 4. 전처리
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
df = df.dropna()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df.drop('Survived', axis=1)
y = df['Survived']

# 5. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. 예측 및 평가
y_pred = model.predict(X_test)

print("📊 분류 리포트:\n")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix 시각화
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. 중요 피처 시각화
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feature_importances.plot(kind='barh')
plt.title("Feature Importance (Random Forest)")
plt.show()
