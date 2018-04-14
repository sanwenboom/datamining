import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Pca:
    def calculate_covariance_matrix(self, input_x, input_y=None):
        """计算协方差矩阵
        :param input:输入矩阵
        :return: 协方差矩阵
        """
        m = input_x.shape[0]
        input_x = input_x - np.mean(input_x, axis=0)
        input_y = input_x if input_y == None else input_y - np.mean(input_y, axis=0)
        return 1 / m * np.matmul(input_x.T, input_y)

    def reducing_dim(self, input, dim):
        """降维到指定维数
        :param input: 输入数据
        :param dim: 目标维数
        :return: 目标维数数据
        """
        covariance_matrix = self.calculate_covariance_matrix(input)
        eigenvalue, eigenvector = np.linalg.eig(covariance_matrix)
        index = eigenvalue.argsort()[::-1]
        eigenvector = eigenvector[:, index]
        eigenvector = eigenvector[:, :dim]
        print(eigenvector)
        result = np.matmul(input, eigenvector)
        return result

def main():
    # 数据集是64维的手写数字数据集，通过PCA降维2个主要成分
    # 鸢尾花数据集由4维降低到2维
    data = datasets.load_digits()
    # data = datasets.load_iris()

    x = data.data
    y = data.target
    transform_data = Pca().reducing_dim(x, 2)
    # 可视化
    x1 = transform_data[:, 0]
    x2 = transform_data[:, 1]
    cmap = plt.get_cmap()
    color_list = [cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
    # color_list = ['pink', 'green', 'black', 'blue', 'yellow', 'purple', 'orange', 'teal', 'wheat', 'red']
    class_data = {}
    class_data['class_name'] = []
    class_data['legend_name'] = []
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        class_data['class_name'].append(plt.scatter(_x1, _x2, color=color_list[i]))
        class_data['legend_name'].append(l)

    plt.legend(class_data['class_name'], class_data['legend_name'], loc=1)
    plt.suptitle("PCA")
    plt.title("Digit")
    plt.xlabel("principal 1")
    plt.ylabel("principal 2")
    plt.show()


if __name__ == '__main__':
    main()