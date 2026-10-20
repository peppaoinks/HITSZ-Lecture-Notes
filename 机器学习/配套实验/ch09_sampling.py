"""实验9.2：反复训练常数预测器，区分拟合误差与总体风险。"""
import numpy as np

def main():
    rng = np.random.default_rng(902)
    for n in [5,20,100]:
        y = rng.normal(2., 3., size=(20000,n))
        c = y.mean(axis=1)
        train = np.mean((y-c[:,None])**2,axis=1)
        population = 9 + (c-2)**2
        print(f'n={n} train={train.mean():.6f} population={population.mean():.6f} '
              f'theory_train={9*(1-1/n):.6f} theory_population={9*(1+1/n):.6f}')
        assert abs(train.mean()-9*(1-1/n)) < .12
        assert abs(population.mean()-9*(1+1/n)) < .12
    print('all checks passed')
if __name__ == '__main__': main()
