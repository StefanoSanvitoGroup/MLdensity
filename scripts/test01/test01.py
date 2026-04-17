from jlgridfingerprints.fingerprints import JLGridFingerprints

settings = {'rcut': 4.08,
            'nmax': [15,6],
            'lmax': 6,
            'alpha': [7.875386069413652,5.875090883472657],
            'beta': [3.6238075908648106,1.7505953204305842],
            'rmin': -0.74,
            'species': ['Al'],
            'body': '1+2',
            'periodic': True,
            'double_shifted': True,
            }

jl = JLGridFingerprints(**settings)

print('Number of JL coefficients: ',jl._n_features)