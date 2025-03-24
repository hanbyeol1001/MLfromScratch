import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#FF0000", '#00FF00', "#0000FF"])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1234)


### 1.데이터의 정보 확인하기
# print(X_train.shape)
# print(X_train[0]) # 훈련샘플

# print(y_train.shape) #120의 라벨
# print(y_train) #3클래스

# plt.figure()
# plt.scatter(X[:,0], X[:,1], c=y , cmap=cmap, edgecolors='k', s=20)
# plt.show()



# #### 2. 최빈값 구하기
# a = [1, 1,1,1,2,2,3,4,5,6]
# from collections import Counter
# most_common = Counter(a).most_common(2)
# #print(most_common) 
# ##출력 결과 : most_common이 1일 때 [(1, 4)]으로 1이 4번 나옴

# print(most_common[0][0]) #튜플 형태 (a,b)의 첫번째 아이템을 반환하며 실제 최빈값 아이템을 확인할 수 있음.



### 3. 이제 Iris 데이터로 knn 확인하기.

from knn import KNN
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)  #how many of our predictions are correctly classified
print("KNN classification accuracy : ", acc)