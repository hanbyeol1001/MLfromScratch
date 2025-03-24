## Logistic Regression - 나는 과연 타이타닉 침몰에서 살아남을 수 있었을까?
# 출처 : https://itstory1592.tistory.com/10
import pandas as pd
import numpy as np

# 사이킷런의 로지스틱 회귀 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#타이타닉 데이터 불러오기
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

#맨 앞의 데이터 10개 출력
print(data.head(10))

#Survived 생존여부를 의미하는 데이터: 0 이면 사망 1이면 생존
#생존 데이터를 타켓으로 사용할 예정

#데이터에 null이 포함되어 있는지 확인
#null값의 데이터는 회귀모델의 분석에 문제를 일으키므로, 모델 훈련시 미리 제거해주는 작업이 필요함.
data.isna().sum()

#출력 결과, 데이터 목록에 null값이 없다고 판단.
#null값이 있었다면, null이 포함된 데이터를 지우거나, 해당 변수의 평균값으로 null값을 채워주는 방법이 있음.

#타겟데이터 따로 저장
target = data["Survived"]

data.drop(labels=["Name","Survived"],axis =1, inplace=True)

#성별 데이터를 숫자로 변환
data["Sex"] = data["Sex"].map({"male":0,"female":1})

data


#훈련데이터와 테스트 데이터 분리
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state=42)

#로지스틱 회귀 인스턴스 생성
lr = LogisticRegression()
#훈련 데이터로 모델 훈련
lr.fit(train_input, train_target)

#예측 결과 출력
print(lr.predict(test_input))

#변수 종류 출력
print(data.head(0))
#각 특징(변수, feature)들의 가중치
print(lr.coef_)

# 성별의 가중치가 높으므로 사망에 가장 큰 영향을 미침을 예상할 수 있음.
# 다음으로 높은 영향력은 "Pclass" -> 이 값은 음수인데도 불구하고 왜 큰 영향을 미쳤냐면
# 높은 등급의 클래스일수록 숫자가 낮으므로, 가중치의 값이 음수더라도 절댓값을 취해서 고려할 필요가 있음.


#과연 나는 타이타닉 침몰 사고에서 살아남을 수 있을까?

#내가 탔을 때를 가정한 조건
#나는 2등석 배에 탔고, 성별은 여자이며, 나이는 29세, 혼자 배에 탔고, 요금을 30.5$를 지불하였음.
pred = lr.predict([[2, 1, 29.0, 0, 0, 30.5789]])

if pred[0]==0 :
    print("AI: 사망할 것으로 예측\n")
else:
    print("AI: 생존할 것으로 예측\n")
    
#음성 클래스 / 양성 클래스의 확률
print("양성 클래스 / 음성클래스 : {}".format(lr.predict_proba([[2, 0, 23.0, 1, 0, 30.5789]])))

#결과 76%의 확률로 생존 