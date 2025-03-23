# 🚀선형 회귀 (Linear Regression)

## 1. 개요
선형 회귀(Linear Regression)는 입력 변수(독립 변수, X)와 출력 변수(종속 변수, Y) 사이의 관계를 선형 함수로 모델링하는 회귀 기법이다. 주어진 데이터를 가장 잘 설명하는 직선을 찾아 새로운 입력 값에 대한 예측을 수행한다.

## 2. 선형 회귀의 종류
### (1) 단순 선형 회귀 (Simple Linear Regression)
단 하나의 독립 변수 $X$를 사용하여 종속 변수 $Y$를 예측하는 모델이다.

$$
Y = wX + b
$$

여기서:
- $ Y $ : 예측값
- $ X $ : 독립 변수 (입력)
- $ w $ : 기울기 (가중치)
- $ b $ : 절편 (bias)

### (2) 다중 선형 회귀 (Multiple Linear Regression)
두 개 이상의 독립 변수 $X_1, X_2, ..., X_n$을 사용하여 종속 변수 $Y$를 예측하는 모델이다.

$$
Y = w_1X_1 + w_2X_2 + ... + w_nX_n + b
$$

여기서 $ w_1, w_2, ..., w_n $은 각 독립 변수에 대한 가중치이다.

## 3. 손실 함수 (Loss Function)
선형 회귀에서는 보통 평균 제곱 오차(Mean Squared Error, MSE)를 손실 함수로 사용한다.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

여기서:
- $ Y_i $ : 실제 값
- $ \hat{Y}_i $ : 예측 값
- $ n $ : 데이터 개수

MSE가 작을수록 예측이 실제 값과 가까움을 의미한다.

## 4. 최적화 방법
### 경사 하강법 (Gradient Descent)
가중치 $ w $와 절편 $ b $를 업데이트하면서 손실 함수를 최소화하는 방법이다.

$$
w = w - \alpha \frac{\partial}{\partial w} MSE
$$

$$
b = b - \alpha \frac{\partial}{\partial b} MSE
$$

여기서:
- $\alpha$ : 학습률 (learning rate)
- $\frac{\partial}{\partial w} MSE$, $\frac{\partial}{\partial b} MSE$ : MSE의 기울기

---

#### 가중치에 대한 그래디언트 계산
손실 함수 $MSE$를 가중치 $W$ 에 대해 편미분하면:

$$
\frac{\partial MSE}{\partial W} = \frac{\partial}{\partial W} \left( \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 \right)
$$

체인 룰을 사용하여 미분하면,

$$
\frac{\partial MSE}{\partial W} = \frac{2}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i) \cdot \frac{\partial (-\hat{Y}_i)}{\partial W}
$$

선형 회귀 모델에서 $\hat{Y} = XW + b$ 이므로,

$$
\frac{\partial \hat{Y}}{\partial W} = X^T
$$

따라서,

$$
\frac{\partial MSE}{\partial W} = -\frac{2}{n} X^T (Y - \hat{Y})
$$

마찬가지로 $b$로 편미분을 진행하면 아래와 같다. 

$$
\frac{\partial MSE}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)
$$


## 5. 가정 (Assumptions)
선형 회귀는 다음과 같은 가정을 기반으로 한다:
1. **선형성 (Linearity)**: 독립 변수와 종속 변수 사이의 관계가 선형이어야 한다.
2. **독립성 (Independence)**: 독립 변수들 간의 다중 공선성이 없어야 한다.
3. **등분산성 (Homoscedasticity)**: 오차(Residuals)의 분산이 일정해야 한다.
4. **정규성 (Normality)**: 오차 항이 정규 분포를 따라야 한다.

## 6. 선형 회귀의 장단점
### 장점
- 구현이 간단하고 해석이 용이하다.
- 계산 비용이 적고 빠르게 학습할 수 있다.
- 데이터가 선형성을 만족할 경우 높은 성능을 보인다.

### 단점
- 독립 변수와 종속 변수 간의 관계가 비선형일 경우 성능이 떨어진다.
- 이상치(Outliers)에 민감하다.
- 다중 공선성(Multicollinearity)이 존재하면 해석이 어려워진다.