###
### GPAW benchmark: Carbon Nanotube
###

from __future__ import print_function
from gpaw.mpi import size, rank
from gpaw.new.ase_interface import GPAW
from gpaw import Mixer, PoissonSolver, ConvergenceError
from gpaw.occupations import FermiDirac

from gpaw.gpu import setup

setup()

try:
    from ase.build import nanotube
except ImportError:
    from ase.structure import nanotube

# dimensions of the nanotube
n = 6
m = 6
length = 10
# other parameters
txt = 'output.txt'
maxiter = 16
conv = {'eigenstates' : 1e-4, 'density' : 1e-2, 'energy' : 1e-3}
# uncomment to use ScaLAPACK
#parallel = {'sl_auto': True}
# uncomment to use GPUs
gpu = {'cuda': True, 'hybrid_blas': False}

# check which GPU backend (if any) is used
if 'gpu' in locals():
    use_cuda = gpu.get('cuda', False)
else:
    use_cuda = False

# output benchmark parameters
if rank == 0:
    print("#"*60)
    print("GPAW benchmark: Carbon Nanotube")
    print("  nanotube dimensions: n=%d, m=%d, length=%d" % (n, m, length))
    print("  MPI tasks: %d" % size)
    print("  using GPU: " + str(use_cuda))
    print("#"*60)
    print("")

# setup parameters
args = {'h': 0.2,
        'nbands': -60,
        'occupations': FermiDirac(0.1),
        'mixer': Mixer(0.1, 5, 50),
#        'poissonsolver': PoissonSolver(eps=1e-12),
        'poissonsolver': PoissonSolver(),
        'eigensolver': 'rmm-diis',
        'maxiter': maxiter,
        'convergence': conv,
        'txt': txt,
        'mode': 'fd'}
if use_cuda:
    parallel = {'gpu': True}
    #args['gpu'] = gpu
    #args['xc_thread'] = False
try:
    args['parallel'] = parallel
except: pass

# setup the system
atoms = nanotube(n, m, length)
atoms.center(vacuum=4.068, axis=0)
atoms.center(vacuum=4.068, axis=1)
calc = GPAW(**args)
atoms.calc=calc
#atoms.set_calculator(calc)

# execute the run
try:
    atoms.get_potential_energy()
except ConvergenceError:
    pass

