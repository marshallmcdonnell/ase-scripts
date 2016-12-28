"""Get energy from a LAMMPS calculation"""

from __future__ import print_function

from ase import Atom
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.visualize import view

from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()

cmds = ["pair_style eam/alloy", "pair_coeff * * NiAlH_jea.eam.alloy Ni H"]

nickel = bulk('Ni', cubic=True)
nickel += Atom('H', position=nickel.cell.diagonal()/2)
# Bit of distortion
nickel.set_cell(nickel.cell + [[0.1, 0.2, 0.4],
                               [0.3, 0.2, 0.0],
                               [0.1, 0.1, 0.1]], scale_atoms=True)

lammps = LAMMPSlib(lmpcmds=cmds,
                   atom_types={'Ni': 1, 'H': 2},
                   log_file='test.log', keep_alive=True)

nickel.set_calculator(lammps)

e = nickel.get_potential_energy()
f = nickel.get_forces()
s = nickel.get_stress()

if me == 0:
        print('Energy: ', e)
        print('Forces:', f)
        print('Stress: ', s)

#view(nickel)
MPI.Finalize()
