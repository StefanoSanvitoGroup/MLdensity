import numpy as np

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

import pickle

from ase.io import read
from pymatgen.io.vasp.outputs import Chgcar

from sklearn import metrics
import glob

def plot_corr_scatter_phases(label,
                             y_true_1h,y_true_1t,y_true_1tp,
                             y_pred_1h,y_pred_1t,y_pred_1tp,
                             savefig=False, save_path=".", model_name="", show=False, hist=False,top_1p=False):

    if not os.path.exists("figures"):
        os.makedirs("figures")

    r2_score = metrics.r2_score(y_true_1tp, y_pred_1tp)
    mae_score = metrics.mean_absolute_error(y_true_1tp, y_pred_1tp)
    rmse_score = metrics.mean_squared_error(y_true_1tp, y_pred_1tp, squared=False)
    
    max_ax = max(y_true_1h.max(),y_true_1t.max(),y_true_1tp.max(),y_pred_1h.max(),y_pred_1t.max(),y_pred_1tp.max()) * 1.2
    min_ax = min(min(y_true_1h.min(),y_true_1t.min(),y_true_1tp.min(),y_pred_1h.min(),y_pred_1t.min(),y_pred_1tp.min()) * 0.6,-0.1 * max_ax)
    if min_ax == 0:
        min_ax = -0.1 * max_ax

    fig, ax = plt.subplots(figsize=(6, 6))
    
    x0, x1 = min_ax, max_ax
    lims = [max(x0, x0), min(x1, x1)]
    ax.plot(lims, lims, c="tab:blue", ls="--", zorder=1, lw=1)

    color_1h = 'tab:red'
    color_1t = 'tab:orange'
    color_1tp = 'tab:blue'

    ax.scatter(y_true_1h ,y_pred_1h,s=49,marker="^" ,color=color_1h ,linewidth=0.05,edgecolors='k',alpha=0.7,label=r'$1H$ (train)',zorder=2)
    ax.scatter(y_true_1t ,y_pred_1t,s=49,marker="v" ,color=color_1t , linewidth=0.05,edgecolors='k',alpha=0.7,label=r'$1T$ (train)',zorder=3)
    ax.scatter(y_true_1tp,y_pred_1tp,s=64,marker="o",color=color_1tp,linewidth=0.05,edgecolors='k',alpha=0.7,label=r'$1T^\prime$ (test)',zorder=1)
               
    ax.set_xlabel("DFT " + label, fontsize=16)
    ax.set_ylabel("ML " + label, fontsize=16)

    unit = r'$e/\rm \AA^3$'

    ax.text(s=r"test R$^2$ =" + f" {r2_score:.6f}",x=0.05 * (max_ax - min_ax) + min_ax,y=0.95 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left",va="top",)
    ax.text(s=f"{'test RMSE':>6} =" + f" {rmse_score:.6f} {unit}",x=0.95 * (max_ax - min_ax) + min_ax,y=0.05 * (max_ax - min_ax) + min_ax,fontsize=16,ha="right",)
    ax.text(s=f"{'test MAE':>6} ="  + f" {mae_score:.6f} {unit}",x=0.95 * (max_ax - min_ax) + min_ax,y=0.125 * (max_ax - min_ax) + min_ax,fontsize=16,ha="right",)
    
    ax.tick_params(labelsize=14, direction="in", top=True, right=True)

    ax.set_xlim((min_ax, max_ax))
    ax.set_ylim((min_ax, max_ax))
    
    ax.legend(fontsize=14,frameon=False,loc='lower right',bbox_to_anchor=(1.0,0.175))

    fig.tight_layout()

    fig.savefig(f'figures/' + model_name + "_scatter.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()

def get_frames_y_values(frames,ml_chgcar_path,partition):
    for io,frame in tqdm(enumerate(frames),total=len(frames)):

        atoms = read(scf_path+f"/{partition}_scf/{frame.replace('-','/')}/POSCAR")
        vol = atoms.get_volume()

        ml_chg = Chgcar.from_file(f'{ml_chgcar_path}/CHGCAR_{frame}').data['total'].ravel() / vol
        dft_chg = Chgcar.from_file(scf_path+f"/{partition}_scf/{frame.replace('-','/')}/CHGCAR").data['total'].ravel() / vol

        if io == 0:
            y_true = np.array(dft_chg,dtype=np.float32)
            y_pred = np.array(ml_chg,dtype=np.float32)
        else:
            y_true = np.append(y_true,dft_chg.astype(np.float32))
            y_pred = np.append(y_pred,ml_chg.astype(np.float32))

    return y_true,y_pred

scf_path = "../../data_scf"
save_path = "chgcar_files"

if not os.path.exists("figures"):
    os.makedirs("figures")

frames_1h = np.sort(np.asarray(['1H_mos2-'+filename.split('/')[-2] for filename in glob.glob(scf_path+'/train_scf/1H_mos2/*/CHGCAR')]))[:1]
frames_1t = np.sort(np.asarray(['1T_mos2-'+filename.split('/')[-2] for filename in glob.glob(scf_path+'/train_scf/1T_mos2/*/CHGCAR')]))[:1]
frames_1tprime = np.sort(np.asarray(['1Tprime_mos2-'+filename.split('/')[-2] for filename in glob.glob(scf_path+'/test_scf/1Tprime_mos2/*/CHGCAR')]))[:1]

print('1H',frames_1h)
print('1T',frames_1t)
print('1Tprime',frames_1tprime)

y_true_1h,y_pred_1h = get_frames_y_values(frames_1h,ml_chgcar_path=save_path,partition='train')
y_true_1t,y_pred_1t = get_frames_y_values(frames_1t,ml_chgcar_path=save_path,partition='train')
y_true_1tp,y_pred_1tp = get_frames_y_values(frames_1tprime,ml_chgcar_path=save_path,partition='test')

plot_corr_scatter_phases(label=r"charge density ($e/\rm \AA^3$)",
                        y_true_1h=y_true_1h,y_true_1t=y_true_1t,y_true_1tp=y_true_1tp,
                        y_pred_1h=y_pred_1h,y_pred_1t=y_pred_1t,y_pred_1tp=y_pred_1tp,
                        savefig=True,model_name=f"phases",show=False,hist=False,top_1p=False)