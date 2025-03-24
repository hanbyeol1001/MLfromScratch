# Data Loadd
from sklearn import datasets

raw_cancer = datasets.load_breast_cancer()

# Feature, Targer
X = raw_cancer.data
y = raw_cancer.target

print(X.shape)
print(y.shape)

print(X[1])
print(y[1])

# Train / Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaling = scaler.transform(X_train)
X_test_scaling = scaler.transform(X_test)

# Training
from sklearn.linear_model import LogisticRegression as LR
LR_model = LR(penalty = 'l2')
LR_model.fit(X_train_scaling, y_train)

# Regression coefficients and intercept
print(LR_model.coef_)
print(LR_model.intercept_)

# Prediction
y_pred = LR_model.predict(X_test_scaling)
print(y_pred)

# Probability
proba_pred = LR_model.predict_proba(X_test_scaling)
print(proba_pred)
print(proba_pred.shape)

# model evaluation
# precision
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(precision)

# confusion matrix
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# classification report
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)