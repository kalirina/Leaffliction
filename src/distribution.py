import sys
import os
from matplotlib import pyplot as plt
import glob
import json


def save_stats():
    stats = {}
    for dir in glob.glob("data/leaves/images/*"):
        if os.path.isdir(dir):
            stats[os.path.basename(dir)] = len(os.listdir(dir))
    with open("data/stats.json", "w") as file:
        json.dump(stats, file, indent=4)


def create_graph(plant):
    path = f"data/leaves/images/{plant}_*"
    data = []
    for dir in glob.glob(path):
        count = len(os.listdir(dir))
        data.append((os.path.basename(dir), count))
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    ax[0].set_title(f"{plant} class distribution")
    ax[0].pie([j for (i, j) in data], colors=colors, autopct='%1.1f%%')
    ax[1].bar([i for (i, j) in data], [j for (i, j) in data], color=colors)
    ax[1].grid(axis='y')
    ax[1].set_axisbelow(True)
    ax[1].set_facecolor("#E5ECF6")
    plt.savefig(f"graphs/{plant}_distribution.png")
    plt.close()


def main():
    args = sys.argv[1:]
    if len(args) > 1:
        print("Too many arguments")
        return
    if len(args) == 0:
        plants = ["Apple", "Grape"]
    else:
        if args[0] not in ["Apple", "Grape"]:
            print("No such plant type")
            return
        plants = [args[0]]
    for plant in plants:
        create_graph(plant)
    save_stats()


if __name__ == '__main__':
    main()

