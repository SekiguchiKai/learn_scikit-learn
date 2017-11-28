# irisのデータセットをimport
from sklearn.datasets import load_iris
# load_irisからirisのデータをロードし、変数に格納
iris_dataset = load_iris()

# irisオブジェクトはk/vで、ここではirisオブジェクト内に存在するkeyの一覧を表示する
print("irisのdatasetsのkeyは: \n{0}".format(iris_dataset.keys()))


# irisオブジェクトの'DESCR'を表示する
print("iris dataset description is as follow.\n {0}".format(iris_dataset['DESCR']))

# 予測する花の名前を確認する
print("Target names : {}".format(iris_dataset['target_names']))

# 特徴量の説明を表示する
print("Faeture names: \n{}".format(iris_dataset['feature_names']))

# データの型を確認する
print("Type of data: {}".format(type(iris_dataset['data'])))

# データの形を確認する
# 出力結果は、 `Shape of data (150, 4)`となる。これは、花の観察データが150、各サンプル1つあたりの特徴量は4つあることになる。
print("Shape of data {}".format(iris_dataset['data'].shape))

print("First 10 columns of data : \n{}".format(iris_dataset['data'][:10]))

print("Last 10 columns of data : \n{}".format(iris_dataset['data'][140:]))

# 計測された花の種類の型を確認する
print("Type of target: {}".format(type(iris_dataset['target'])))

#
print("Shape of target: {}".format(iris_dataset['target'].shape))

# エンコードされた花の種類を確認する
# 0: 'setosa' 1: 'versicolor' 2:'virginica'
print("Target: \n{}".format(iris_dataset['target']))

