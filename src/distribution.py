import sys
import os
from matplotlib import pyplot as plt
import glob


def main():
    if len(sys.argv[1:]) != 1:
        raise ValueError("Too many arguments")
    path = "data/leaves/images/" + sys.argv[1] + "_*"
    if sys.argv[1] != "Apple" and sys.argv[1] != "Grape":
        raise ValueError("No such plant type")
    data = []
    for dir in glob.glob(path):
        count = len(os.listdir(dir))
        data.append((os.path.basename(dir), count))
    fig, ax = plt.subplots(1, 2, figsize=(15,8))
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    ax[0].set_title(sys.argv[1] + " class distribution")
    ax[0].pie([j for (i,j) in data], colors=colors, autopct='%1.1f%%')
    ax[1].bar([i for (i,j) in data], [j for (i,j) in data], color=colors)
    ax[1].grid(axis='y')
    ax[1].set_axisbelow(True)
    ax[1].set_facecolor("#E5ECF6")
    plt.savefig("graphs/" + sys.argv[1] + "_distribution.png")


if __name__ == '__main__':
    main()

