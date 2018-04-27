import numpy as np
from sklearn import datasets

class Data:
    def __init__(self, features, label):
        """
        :param features: list<float>
        :param label: int
        """
        self.features = features
        self.label = label
class Knn:
    def __init__(self, data, k=10):
        """
        :param data: list<Data>
        :param k: int
        """
        self.k = k
        self.data = data


    def calaculate_dist(self, input):
        """
        :param input: list<float>
        :return: int
        """
        dist_list = []
        for each in self.data:
            dist = np.sqrt(np.sum(np.square(each.features - input)))
            label = each.label
            dict = {"dist":dist, "label":label}
            dist_list.append(dict)
            dist_list = sorted(dist_list, key=lambda x: x["dist"])
            print(dist_list)
        return dist_list[:10]

    def calculate_label(self, dist_list):
        """
        :param dist_list: list<{"dist":float, "label":int}>
        :return: int
        """
        print(dist_list)
        dict = {}
        for each in dist_list:
            if each["label"] in dict.keys():
                dict[each["label"]] += 1
            else:
                dict[each["label"]] = 1
        result = sorted(zip(dict.values(),dict.keys()))[0]
        return result[1]
    def process(self, input):
        return self.calculate_label(self.calaculate_dist(input))
def main():
    data = datasets.load_iris()
    features = data.data

    labels = data.target
    print(labels)
    knn_data = []
    for i, feature in enumerate(features):
        each_data = Data(feature, labels[i])
        knn_data.append(each_data)
    input = [5.1, 3.5, 1.4, 0.2]
    print(Knn(knn_data).process(input))


if __name__ == '__main__':
    main()
