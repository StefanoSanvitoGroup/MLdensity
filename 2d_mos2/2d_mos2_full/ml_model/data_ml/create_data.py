import numpy as np

import sys
sys.path.append('../../../')
sys.path.append('../../../jlgridfingerprints')
from jlgridfingerprints.fingerprints import JLGridFingerprints
from jlgridfingerprints.tools import create_grid_coords,sample_charge

from ase.io import read
from pymatgen.io.vasp.outputs import Chgcar
import h5py

from sklearn.utils import shuffle

from tqdm import tqdm
import time

import os
import glob

from numpy.random import default_rng
rng = default_rng(42)

def write_chg_files(data, partition, frame=None):
    if not frame is None:
        h5f = h5py.File(f"y_{partition}_data_{frame}.h5", "w")
    else:
        h5f = h5py.File(f"y_{partition}_data.h5", "w")
    dset = h5f.create_dataset("chg", data=data, compression="gzip", compression_opts=9)
    h5f.close()
    del dset

def write_desc_files(data, partition, frame=None):

    if not frame is None:
        h5f = h5py.File(f"X_{partition}_data_{frame}.h5", "w")
    else:
        h5f = h5py.File(f"X_{partition}_data.h5", "w")

    dset = h5f.create_dataset("desc", data=data, compression="gzip", compression_opts=9)
    h5f.close()
    del dset

settings = {'rcut': 4.76,
            'nmax': [18,11],
            'lmax': 10,
            'alpha': [6.72,5.07],
            'beta': [6.97,2.69],
            'rmin': -0.93,
            'species': ['Mo','S'],
            'body': '1+2',
            'periodic': [True,True,False],
            'double_shifted': True,
            }

jl = JLGridFingerprints(**settings)

print('Number of JL coefficients: ',jl._n_features)

sigma = 40.0
uniform_ratio = 0.60
n_samples_ratio = 0.05 / 100.0

scf_path = "../../data_scf"

frames_1h = np.sort(np.asarray(['1H_mos2-'+filename.split('/')[-2] for filename in glob.glob(scf_path+'/train_scf/1H_mos2/*/CHGCAR')]))
frames_1t = np.sort(np.asarray(['1T_mos2-'+filename.split('/')[-2] for filename in glob.glob(scf_path+'/train_scf/1T_mos2/*/CHGCAR')]))
frames_1tprime = np.sort(np.asarray(['1Tprime_mos2-'+filename.split('/')[-2] for filename in glob.glob(scf_path+'/test_scf/1Tprime_mos2/*/CHGCAR')]))
rng.shuffle(frames_1h)
rng.shuffle(frames_1t)
rng.shuffle(frames_1tprime)

ntrain_h = 10
ntrain_t = 10
ntest = 10

frames_dict = {'train': np.append(frames_1h[:ntrain_h],frames_1t[:ntrain_t]), 'test': frames_1tprime[:ntest]}
print('train_frames:', frames_dict['train'])
print('test_frames:', frames_dict['test'])

time_descriptor = 0.0
time_open = 0.0
time_write = 0.0

X_train_data = []
y_train_data = []
X_test_data = []
y_test_data = []
with tqdm(total=sum([len(frames_dict[partition]) for partition in ['train','test']])) as pbar:
    for partition in ['train','test']:
        for frame in frames_dict[partition]:

            t_init = time.time()
            atoms = read(scf_path + f"/{partition}_scf/{frame.replace('-','/')}/POSCAR")
            chgcar = Chgcar.from_file(scf_path + f"/{partition}_scf/{frame.replace('-','/')}/CHGCAR")
            time_open += (time.time() - t_init)

            vol = atoms.get_volume()

            ngxf,ngyf,ngzf = chgcar.data['total'].shape

            n_samples_per_snapshot = int(np.ceil((ngxf*ngyf*ngzf) * n_samples_ratio))

            fcoords = create_grid_coords(grid_size=chgcar.data["total"].shape,return_cartesian_coords=False)
            chg_points = chgcar.data["total"].ravel() / vol

            selected_index = sample_charge(chg=chg_points,sigma=sigma,n_samples=n_samples_per_snapshot,uniform_ratio=uniform_ratio,seed=42)

            fcoords = fcoords[selected_index]
            cart_coords = np.dot(fcoords, atoms.get_cell().array)
            chg_points = chg_points[selected_index]

            t_init = time.time()
            jl_points = jl.create(atoms, cart_coords)
            time_descriptor += (time.time() - t_init)

            t_init = time.time()
            if partition == 'train':
                X_train_data.append(jl_points)
                y_train_data.append(chg_points)
            elif partition == 'test':
                X_test_data.append(jl_points)
                y_test_data.append(chg_points)
            time_write += (time.time() - t_init)

            pbar.update(1)

X_train_data = np.vstack(X_train_data)
y_train_data = np.hstack(y_train_data)

X_test_data = np.vstack(X_test_data)
y_test_data = np.hstack(y_test_data)

n_frames = sum([len(frames_dict[partition]) for partition in ["train", "test"]])
time_descriptor /= n_frames
time_open /= n_frames
time_write /= n_frames

print(f'JL coeff    : {time_descriptor:>5.3f} sec for {n_samples_per_snapshot} points over {n_frames} structures')
print(f'Open files  : {time_open:>5.3f} sec')
print(f'Write files : {time_write:>5.3f} sec')

X_train_data, y_train_data = shuffle(X_train_data, y_train_data, random_state=42)

write_desc_files(X_train_data, "train")
write_desc_files(X_test_data, "test")

write_chg_files(y_train_data, "train")
write_chg_files(y_test_data, "test")