import matplotlib.pyplot as plt
import numpy as np

# 粒子类
class Partical:
    def __init__(self):
        self.pos = 0
        self.speed = 0
        self.pbest = 0
        self.fitness = 0

# 粒子群算法
class PSO:
    def __init__(self):
        self.w = 0.5
        self.c1 = 1
        self.c2 = 2
        self.gbest = 0
        self.num = 10
        self.POP = []
        self.iterNum = 1000
    #评价函数
    def fitness(self, x):
        return  x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)
    # 全局最优
    def gbestSearch(self, pop):
        for partical in pop:
            if partical.fitness > self.fitness(self.gbest):
                self.gbest = partical.pos
    # 初始化粒子群
    def initPopulation(self, pop, N):
        for i in range(N):
            partical = Partical()
            partical.pos = np.random.uniform(-10, 10)
            partical.fitness = self.fitness(partical.pos)
            partical.pbest = partical.fitness
            pop.append(partical)
        self.gbestSearch(pop)
    # 粒子更新
    def update(self, pop):
        for partical in pop:
            speed = self.w * partical.speed + self.c1*np.random.random()*(partical.pbest-partical.pos)+self.c2*np.random.random()*(self.gbest-partical.pos)

            pos = partical.pos + speed

            if -10 < pos < 10:
                print(pos)
                partical.speed = speed
                partical.pos = pos
                partical.fitness = self.fitness(partical.pos)
                if partical.fitness > self.fitness(partical.pbest):
                    partical.pbest = partical.pos
    # 迭代
    def start(self):
        self.initPopulation(self.POP, self.num)

        for i in range(self.iterNum):
            self.update(self.POP)

            self.gbestSearch(self.POP)


# 可视化
pso = PSO()
pso.start()
outputx = []
outputy = []

for partical in pso.POP:
    outputx.append(partical.pos)
    outputy.append(partical.fitness)
    print("x=", partical.pos, "f(x)=", partical.fitness)
print(pso.gbest, pso.fitness(pso.gbest))
x = np.linspace(-10, 10, 10000)
y = x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)
plt.plot(x, y)
plt.scatter(outputx, outputy)
plt.show()




