from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. 데이터 준비 (X, y는 이미 있다고 가정)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 강력한 모델(Teacher) 학습
# n_estimators: 트리의 개수 (많을수록 강력해짐)
# max_depth: 각 트리의 깊이 (너무 깊지 않게 설정)
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob', # 다중 클래스 분류
    random_state=42
)

xgb_model.fit(X_train, y_train)

# 3. 성능 평가
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

print(f"XGBoost 학습 정확도: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"XGBoost 테스트 정확도: {accuracy_score(y_test, y_pred_test):.4f}")
