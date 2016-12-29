"""Get energy from a LAMMPS calculation"""

from __future__ import print_function

from ase import units
from ase.calculators.lammpslib import LAMMPSlib, write_lammps_data
from ase.neighborlist import NeighborList
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.visualize import view
from PawleySF6 import PawleySF6


if __name__ == "__main__":

    # Set up SF6 system
    sf6 = PawleySF6(supercell=[1,1,1])
    header, cmds = sf6.lammpsParameters()
    molecules = sf6.system
    
    # Set momenta corresponding to 300K

    # parallel via mpi4py
    try:
        from mpi4py import MPI
        me = MPI.COMM_WORLD.Get_rank()
    except:
        me = 0

    # Set LAMMPS as Calculator
    lammps = LAMMPSlib(lmpcmds=cmds,
                       lammps_header=header,
                       atom_types={'S': 1, 'F': 2},
                       read_molecular_info=True,
                       log_file='test.log', keep_alive=True)

    molecules.set_calculator(lammps)

    write_lammps_data('sf6.data', 
                      molecules, 
                      {'S': 1, 'F': 2}, 
                      cutoff=1.6,
                      bond_types=[(16,9)],
                      angle_types=[(9,16,9)],
                      units='real')

    view(molecules)
                    
    exit()

    # Get initial energy, force, and stress

    energy = molecules.get_potential_energy()
    forces = molecules.get_forces()
    stress = molecules.get_stress()

    if me == 0:
            print('Energy: ', energy)
            print('Forces:', forces)
            print('Stress: ', stress)

    # Simulation run

    MaxwellBoltzmannDistribution(molecules, 5000.*units.kB)
    dyn = VelocityVerlet(molecules, dt=1.0 * units.fs,
                         logfile='md.log')

    def printenergy(a=molecules):  # store a reference to atoms in the definition.
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        if me == 0:
            print('Energy per atom: Epot = %.3f kcal/mol  Ekin = %.3f kcal/mol (T=%3.0fK)  '
              'Etot = %.3f kcal/mol' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

    dyn.attach(printenergy, interval=10)
    traj=Trajectory('md.traj', 'w', molecules)
    dyn.attach(traj.write, interval=10)

    printenergy()
    #dyn.run(100)

    MPI.Finalize()
