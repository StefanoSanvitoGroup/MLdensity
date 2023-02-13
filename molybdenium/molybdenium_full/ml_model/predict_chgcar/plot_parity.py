import numpy as np

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

import pickle

from ase.io import read
from pymatgen.io.vasp.outputs import Chgcar

from sklearn import metrics

def plot_corr_scatter(label, y_true, y_pred, savefig=False, save_path=".", model_name="", show=False):

    r2_score = metrics.r2_score(y_true, y_pred)
    mae_score = metrics.mean_absolute_error(y_true, y_pred)
    mse_score = metrics.mean_squared_error(y_true, y_pred)
    rmse_score = metrics.mean_squared_error(y_true, y_pred, squared=False)
    maxae_score = metrics.max_error(y_true, y_pred)

    max_ax = max(y_true.max(),y_pred.max()) * 1.2
    min_ax = min(min(y_true.min(),y_pred.min()) * 0.6,-0.1 * max_ax)
    
    if min_ax == 0:
        min_ax = -0.1 * max_ax

    fig, ax = plt.subplots(figsize=(6, 6))

    x0, x1 = min_ax, max_ax
    lims = [max(x0, x0), min(x1, x1)]
    ax.plot(lims, lims, c="tab:blue", ls="--", zorder=1, lw=1)

    ax.scatter(y_true,y_pred,marker="o",color="tab:blue",linewidth=0.05,edgecolors='k',alpha=1.0,)

    ax.set_xlabel("DFT " + label, fontsize=16)
    ax.set_ylabel("ML " + label, fontsize=16)

    ax.text(s=r"R$^2$ =" + f" {r2_score:.6f}",x=0.05 * (max_ax - min_ax) + min_ax,y=0.95 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left",va="top")
    ax.text(s=r"RMSE  =" + f" {rmse_score:.6f} "+r"e/$\rm \AA^3$",x=0.95 * (max_ax - min_ax) + min_ax,y=0.05 * (max_ax - min_ax) + min_ax,fontsize=16,ha="right")
    ax.text(s=r"MAE   =" + f" {mae_score:.6f} "+r"e/$\rm \AA^3$",x=0.95 * (max_ax - min_ax) + min_ax,y=0.125 * (max_ax - min_ax) + min_ax,fontsize=16,ha="right")
    # ax.text(s=r"MaxAE =" + f" {maxae_score:.6f}",x=0.5 * (max_ax - min_ax) + min_ax,y=0.25 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left")
    
    ax.tick_params(labelsize=14, direction="in", top=True, right=True)

    ax.set_xlim((min_ax, max_ax))
    ax.set_ylim((min_ax, max_ax))

    # ax.set_title(model_name, fontsize=14, ha="center")

    fig.tight_layout()

    if savefig:
        fig.savefig(save_path + "/" + model_name + "_scatter.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

scf_path = "../../data_scf"
save_path = "chgcar_files"
frames = np.arange(10,20,dtype=int)
nframes = len(frames)

if not os.path.exists("figures"):
    os.makedirs("figures")

time_open = 0.0
time_plot = 0.0

dft_chg = []
ml_chg = []
for frame in tqdm(frames,total=nframes):

    t_init = time.time()
    atoms = read(f"{scf_path}/test_scf/{frame}/POSCAR")
    dft_chgcar = Chgcar.from_file(f"{scf_path}/test_scf/{frame}/CHGCAR")
    ml_chgcar = Chgcar.from_file(f"{save_path}/CHGCAR_{frame}")
    time_open += (time.time() - t_init)

    vol = atoms.cell.volume

    dft_chg.append(dft_chgcar.data['total'].ravel() / vol)
    ml_chg.append(ml_chgcar.data['total'].ravel() / vol)

dft_chg = np.hstack(dft_chg)
ml_chg = np.hstack(ml_chg)

test_r2 = metrics.r2_score(dft_chg, ml_chg)
test_mae = metrics.mean_absolute_error(dft_chg, ml_chg)
test_mse = metrics.mean_squared_error(dft_chg, ml_chg)
test_rmse = metrics.mean_squared_error(dft_chg, ml_chg, squared=False)
test_maxae = metrics.max_error(dft_chg, ml_chg)

print(f"           |        R2       |      RMSE       |       MAE       |      MaxAE      |       MSE     ",flush=True,)
print(f" All       |  {test_r2: 8.6e}  |  {test_rmse: 8.6e}  |  {test_mae: 8.6e}  |  {test_maxae: 8.6e}  |  {test_mse: 8.6e}  ",flush=True,)

t_init = time.time()
plot_corr_scatter(r"charge density (e/$\rm \AA^3$)",dft_chg,ml_chg,savefig=True,save_path="figures",model_name=f"test_all",show=True)
time_plot += (time.time() - t_init)

time_open /= nframes
time_plot /= nframes

print(f'Open files  : {time_open:>5.3f} sec')
print(f'Plot files  : {time_plot:>5.3f} sec')