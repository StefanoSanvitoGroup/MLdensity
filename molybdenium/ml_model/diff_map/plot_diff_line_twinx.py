import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.vasp.outputs import Chgcar
from ase.io import read

import sys
sys.path.append('../../../')
sys.path.append('../../../jlgridfingerprints')
from jlgridfingerprints.fingerprints import JLGridFingerprints

import time
import os
from tqdm import tqdm
import pickle

settings = {'rcut': 4.04,
            'nmax': [20,12],
            'lmax': 11,
            'alpha': [4.016939668269249,-0.0827444142560686],
            'beta': [5.4622394140294785,2.378398890338807],
            'rmin': -1.09,
            'species': ['Mo'],
            'body': '1+2',
            'periodic': True,
            'double_shifted': True,
            }

jl = JLGridFingerprints(**settings)

print('Number of JL coefficients: ',jl._n_features)

scf_path = "../../data_scf"
model_name = 'linear_model_chg'

frame = 13

atoms = read(scf_path + f"/test_scf/{frame}/POSCAR")
vol = atoms.get_volume()

chgcar = Chgcar.from_file(scf_path + f"/test_scf/{frame}/CHGCAR")

ngxf,ngyf,ngzf = chgcar.data['total'].shape
y_plane = 51
z_plane = 0.0

frac_points_quarter = np.zeros((ngxf,3))

chg_dft_points_quarter = np.zeros((ngxf,1))

io = 0
for nx in range(ngxf):
    frac_points_quarter[io] = (nx/ngxf,y_plane/ngyf,z_plane)
    chg_dft_points_quarter[io] = chgcar.data['total'][nx,y_plane,int(z_plane*ngzf)] / vol
    io += 1

cart_positions_quarter = np.dot(frac_points_quarter,atoms.get_cell().array)

descriptors_quarter = jl.create(atoms, cart_positions_quarter)

model = pickle.load(open(f'../train_ml/scikit_{model_name}.p','rb'))

chg_pred_points_quarter = model.predict(descriptors_quarter).reshape((-1,1))

chg_diff_points_quarter = chg_pred_points_quarter - chg_dft_points_quarter

fig, ax = plt.subplots(figsize=(8,5.0175),dpi=100)
dft_handle = ax.plot(np.arange(ngxf),chg_dft_points_quarter.ravel(),lw=3,color='navy',ls='-',label='DFT')
ml_handle = ax.plot(np.arange(ngxf),chg_pred_points_quarter.ravel(),lw=2,color='darkorange',ls='--',label='ML')
ax.set_xlim((0,ngxf-1))
ax.set_xticks([tick for tick in range(0,ngxf,20)])
ax.tick_params(labelsize=14,direction='in',top=True,left=True,bottom=True,right=False)
ax.set_ylabel(r'Charge density ($e/\rm \AA^3$)',fontsize=16)
ax.set_xlabel(r'Grid points along x',fontsize=16)
ax.set_ylim([-2.5,11])
error_ax = ax.twinx()  # instantiate a second axes that shares the same x-axis
error_ax.set_ylabel(r'ML$-$DFT ($e/\rm \AA^3$)', color='red',fontsize=16)  # we already handled the x-label with ax1
error_handle = error_ax.plot(np.arange(ngxf),chg_diff_points_quarter.ravel(),lw=2,color='red',ls='-.',label=r'ML$-$DFT')
error_ax.tick_params(axis='y', labelcolor='red',labelsize=14,direction='in',color='red')
error_ax.spines['right'].set_edgecolor('red')
error_ax.set_ylim([ax.get_ylim()[0]/100,ax.get_ylim()[1]/100])
fig.tight_layout()
legend_handles = dft_handle+ml_handle+error_handle
legend_labels = [l.get_label() for l in legend_handles]
ax.legend(legend_handles,legend_labels,fontsize=14,frameon=False,ncol=3,loc='upper right')
fig.savefig('figures/'+f'line_diff_mo_{frame}_y{y_plane}.png',dpi=600,bbox_inches='tight')
plt.show()