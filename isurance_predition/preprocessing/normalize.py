import math


class Normalize:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mean_values = []
        self.std_values = []
        self.min_values = []
        self.max_values = []
        self.handle()

    def handle(self):
        for i in range(len(self.dataset[0])):
            self.dataset.sort(key=lambda tup: tup[i])
            self.min_values.append(self.dataset[0][i])
            self.max_values.append(self.dataset[-1][i])

        for i in range(len(self.dataset[0])):
            values = list()
            for data_point in self.dataset:
                values.append(data_point[i])
            self.mean_values.append(sum(values)/len(values))

        for i, m in enumerate(self.mean_values):
            tmp = list()
            for data_point in self.dataset:
                tmp.append(pow((data_point[i]-m), 2))
            self.std_values.append(math.sqrt(sum(tmp)/len(self.dataset)))

    def minmax(self, predict_point=None):
        if predict_point is not None:
            for i in range(len(predict_point)):
                data_point[i] = (data_point[i] - self.min_values[i]) / (self.max_values[i] - self.min_values[i])
            return predict_point
        else:
            for data_point in self.dataset:
                for i in range(len(data_point)):
                    data_point[i] = (data_point[i] - self.min_values[i]) / (self.max_values[i] - self.min_values[i])
            return self.dataset

    def standardization(self, predict_point=None):
        if predict_point is not None:
            for i in range(len(predict_point)):
                data_point[i] = (
                    data_point[i] - self.mean_values[i]) / self.std_values[i]
            return predict_point
        else:
            for data_point in self.dataset:
                for i, value in enumerate(data_point):
                    data_point[i] = (value - self.mean_values[i]) / self.std_values[i]
            return self.dataset
