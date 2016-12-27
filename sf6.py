"""Get energy from a LAMMPS calculation"""

from __future__ import print_function

from ase import Atoms, units
from ase.calculators.lammpslib import LAMMPSlib
from ase.constraints import FixBondLengths, FixInternals 
from ase.md.verlet import VelocityVerlet
from ase.visualize import view

# Constrain SF6 molecule - make rigid molecule
def constrainSF6( molecule, shift_index=0 ):

	# Get SF bond list
	s_index   = [x.index for x in molecule if x.symbol == 'S'][0]
	f_indices = [x.index for x in molecule if x.symbol == 'F']

	sf_bonds = list()
	for f_index in f_indices:
	  sf_bond = [s_index+shift_index, f_index+shift_index]
	  #sf_bonds.append( [molecule.get_distance(s_index, f_index), sf_bond] )
	  sf_bonds.append(sf_bond)

	# Get FSF angle list
	fsf_angles = list()
	for f1 in f_indices:
	  for f2 in f_indices:
	    if f1 < f2:
	      fsf_angle = [f1+shift_index, s_index+shift_index, f2+shift_index]
	      fsf_angles.append( [molecule.get_angle([f1,s_index,f2]), fsf_angle] )


	#molecular_constraints = FixInternals( bonds=sf_bonds, angles=fsf_angles )
	molecular_constraints = FixBondLengths( sf_bonds )
	return molecular_constraints


# Create SF6 molecule via Pawley model (G. Pawley, Mol. Phys. 43 (1981) 1321.) 

d = 2.89
l = 1.565

molecules = Atoms('S', positions=[(l,l,l)], cell=[2.*d,2.*d,2.*d], pbc=[1,1,1])
molecules += Atoms('6F', 
	     positions=[(l,l,0.),
			(l,0.,l),
			(0.,l,l),
			(l,l,2.*l),
			(l,2.*l,l),
			(2.*l,l,l)]
            )


# Constrain SF6 molecule - make rigid molecule
mol1_constraints = constrainSF6(molecules)

# Create 2nd SF6 molecule

mol2 = molecules.copy()
mol2.translate([d,d,d])
mol2_constraints = constrainSF6(mol2,shift_index=7)

molecules+=mol2

molecular_constraints = [mol1_constraints, mol2_constraints]
molecules.set_constraint(molecular_constraints)

# Simulation Parameters

eps_ff = 70.60 # units: K
sig_ff =  2.70 # units: Angstrom

eps_ff *= units.kB / (units.kcal/units.mol) # units: kcal/mol

header = ['units real', 
          'atom_style molecular',
          'atom_modify map array sort 0 0']

cmds = ["pair_style lj/cut 10.0", 
	"pair_coeff * * 0.0 0.0",
	"pair_coeff 2 2 "+str(eps_ff)+' '+str(sig_ff)]

# parallel via mpi4py

from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

lammps = LAMMPSlib(lmpcmds=cmds,
                   atom_types={'S': 1, 'F': 2},
                   log_file='test.log', keep_alive=True)

molecules.set_calculator(lammps)


# Get initial energy, force, and stress

print('Energy: ', molecules.get_potential_energy())
print('Forces:', molecules.get_forces())
print('Stress: ', molecules.get_stress())

# Simulation run

dyn = VelocityVerlet(molecules, dt=1.0 * units.fs,
                     trajectory='md.traj', logfile='md.log')
dyn.run(10)

MPI.Finalize()
