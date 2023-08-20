from src.models.bamvc.minibatch import train_main
import matplotlib.pyplot as plt
from pathlib import Path as P
import logging
import numpy as np
import warnings
import itertools
from src.vis.visualize import *
from src.utils.io_utils import *
from src.utils.torch_utils import *

warnings.filterwarnings(action="ignore")
logging.basicConfig(level=logging.INFO)

do_vis = True

if __name__ == "__main__":
    method = "minibatch"
    dataname = "ORL-40"
    datapath = P("./data").joinpath(dataname)
    savedir = P(f"../output/debug/{method}/{dataname}")

    train_main(
        datapath=datapath,
        method=method,
        savedir=savedir,
        save_vars=True,
        num_bags=10,
        epochs=100,
        lr=0.1,
    )

    if not do_vis:
        exit()

    H_common = load_var(savedir, "H_common")
    sns.heatmap(pairwise_distances(H_common))
    plt.show()

    plot_scatters(H_common, datapath=datapath, palette="hls")
    plt.show()

    Z_common = load_var(savedir, "Z_common")
    sns.heatmap(Z_common)
    plt.show()
