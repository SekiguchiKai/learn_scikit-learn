# irisのデータセットをimport
from sklearn.datasets import load_iris
# load_irisからirisのデータをロードし、変数に格納
iris_dataset = load_iris()

# irisオブジェクトはk/vで、ここではirisオブジェクト内に存在するkeyの一覧を表示する
print("irisのdatasetsのkeyは: \n{0}".format(iris_dataset.keys()))


# irisオブジェクトの'DESCR'を表示する
print("iris dataset description is as follow.\n {0}".format(iris_dataset['DESCR']))





