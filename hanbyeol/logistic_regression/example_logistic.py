import argparse
import sys
import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from logistic_regression import LogisticRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import accuracy

def main():
    parser = argparse.ArgumentParser(description="Logistic Regression Example")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["breast_cancer", "iris", "synthetic"], 
        default="breast_cancer",
        help="Dataset choice: 'breast_cancer', 'iris', or 'synthetic'"
    )
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--n_iters", type=int, default=1000, help="Number of training iterations (default: 1000)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # 데이터셋 로드
    print("📥 데이터셋 로드 중...")
    if args.dataset == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        task_name = "Breast Cancer Classification"
    elif args.dataset == "iris":
        data = load_iris()
        X, y = data.data, (data.target == 2).astype(int)  # binary classification: class 2 vs rest
        task_name = "Iris (class 2 vs rest)"
    else:
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=args.random_state)
        task_name = "Synthetic Classification"
    print(f"✅ 데이터 로드 완료: {args.dataset}, X={X.shape}, y={y.shape}")

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 모델 학습
    print("📚 모델 훈련 중...")
    model = LogisticRegression(learning_rate=args.learning_rate, n_iters=args.n_iters)
    model.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)

    # 예측 확률 얻기 (ROC용)
    y_scores = 1 / (1 + np.exp(-(np.dot(X_test, model.weights) + model.bias)))

    print(f"✅ {task_name} 정확도: {acc:.4f}")

    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve: {task_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
