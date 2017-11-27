from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

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

