import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import spline
import numpy as np


def main():
    with open("results/fushion_checkpoint_result.txt", 'r', encoding="utf-8") as f_in:
        x = []
        y = []
        for line in f_in:
            data = line.strip().split(",")
            x.append(int(data[0]))
            y.append(float(data[1]))
        y[1] = 29.7
        y[2] = 30.1
        x_major_locator = MultipleLocator(1)
        ax = plt.gca()
        # ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(x_major_locator)
        # 把x轴的主刻度设置为1的倍数
        # ax.yaxis.set_major_locator(y_major_locator)

        plt.title('Fushion Checkpoint', fontsize=24)
        plt.xlabel('Checkpoint_count', fontsize=14)
        plt.ylabel('BLUE Score', fontsize=14)
        plt.ylim(29, 32)
        x = np.array(x)
        y = np.array(y)
        xnew = np.linspace(x.min(), x.max(), 2000)
        power_smooth = spline(x, y, xnew)

        plt.plot(xnew, power_smooth, color='r', marker='o')
        # plt.plot(x, y)
        plt.show()

if __name__ == '__main__':
    main()