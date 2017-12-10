import numpy as np
import matplotlib.pyplot as plt


class Individual:
    def __init__(self):

        self.x = 0
        self.evaluation = 0
        self.selection = 0
        self.fitness = 0


class GA:
    def __init__(self):
        self.iterNum = 10000
        self.N = 20
        self.pop = []
        self.w = 0.9
        self.r1 = 0.75
        self.r2 = 0.1

    def evaluate(self, x):
        return x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)

    def fit(self, pop):
        evalSum = 0
        for individual in pop:
            evalSum += individual.evaluation
        average = evalSum/len(pop)
        for individual in pop:
            individual.fitness = individual.evaluation/average
            individual.selection = individual.evaluation/abs(evalSum)

    def initPop(self, n):
        for i in range(n):
            individual = Individual()
            individual.x = np.random.uniform(-10,10)
            individual.evaluation = self.evaluate(individual.x)
            self.pop.append(individual)

    def select(self):
        self.pop.sort(key=lambda x: -x.selection)
        select1 = np.random.random_integers(0, self.N-1)
        select2 = np.random.random_integers(0, self.N-1)
        while select1 == select2:
            select2 = np.random.random_integers(0, self.N-1)
        cross1 = self.w*self.pop[select1].x +(1-self.w)*self.pop[select2].x
        cross2 = self.w * self.pop[select2].x + (1 - self.w) * self.pop[select1].x
        new = sorted([self.pop[select1].x, self.pop[select2].x, cross1, cross2], key=lambda x: -self.evaluate(x))
        self.pop[select1].x = new[0]
        self.pop[select2].x = new[1]
        self.pop[select1].evaluation = self.evaluate(self.pop[select1].x)
        self.pop[select2].evaluation = self.evaluate(self.pop[select2].x)

    def mutate(self):
        select = np.random.random_integers(0, self.N-1)
        self.pop[select].x = np.random.uniform(-10, 10)
        self.pop[select].evaluatiom = self.evaluate(self.pop[select].x)

    def start(self):
        self.initPop(self.N)
        for i in range(self.iterNum):
            self.fit(self.pop)
            print(self.pop[0].x, self.pop[0].evaluation)
            if np.random.random() < self.r1:
                self.select()
            if np.random.random() < self.r2:
                self.mutate()


if __name__ == '__main__':
    ga = GA()
    ga.start()
    outputx = []
    outputy = []

    for partical in ga.pop:
        outputx.append(partical.x)
        outputy.append(partical.evaluation)
    x = np.linspace(-10, 10, 10000)
    y = x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)
    plt.plot(x, y)
    plt.scatter(outputx, outputy)
    plt.show()



