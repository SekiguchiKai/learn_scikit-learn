from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

print("=====データを準備する======")
# load_irisからirisのデータをロードし、変数に格納
iris_dataset = load_iris()

# train_test_split()を使用してデータを訓練データ: 75%、テストデータ: 25%に分ける
# 入力データをX、ラベルをyで表す
# また、2次元配列のデータはX、1次元配列であるラベルはyで表す
# ここではシャッフルしている
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

print("X_train shape: {}".format(X_train.shape))
print("X_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("X_test shape: {}".format(y_test.shape))

# 現在Num_Py配列になっているX_trainをpandasのDataframeに変換する
# iris_dataset.feature_namesから、pandasのDataFrameのcolumnに名前を付与する
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

print("=====k-最近傍法を用いてモデルを生成する======")
# k-最近傍法
# データポイントを訓練セットの中から一番近い点を探して、新しいデータに付与する
# 近傍点を設定して、インスタンス化
knn = KNeighborsClassifier(n_neighbors=1)
# 訓練データを引数にして、モデルを作成
knn.fit(X_train, y_train)
print(knn)

print("=====k-最近傍法を用いて予測を行う======")
# 新しいirisのデータを作成し、予測してみる
print("新しいirisのデータを作成し、予測してみる")
# 新しいirisのデータをnimPyの配列として作成
# 4つの特徴量
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_train.shape))

# 予測を行う
# knn.predictで予測を行う
prediction = knn.predict(X_new)

print("予測結果")
print("Prediction: {}".format(prediction))
# Predictionは0なのでこう言う書き方できる
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

print("=====モデルの評価を行う======")