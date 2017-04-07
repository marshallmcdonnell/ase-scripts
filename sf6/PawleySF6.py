"""Pawley SF6 molecule model"""

from __future__ import print_function

from ase import Atoms, units
from ase.constraints import FixInternals
from ase.visualize import view


# Create SF6 molecule via Pawley model (G. Pawley, Mol. Phys. 43 (1981) 1321.) 
class PawleySF6(object):

    def __init__(self, sfbond_distance=1.565, lat=5.780, supercell=[1,1,1]):
        self._sfbond_distance = sfbond_distance
        self._lat = lat
        self._supercell = supercell

        self.molecules = list()
        self.unitCell = None

        # make SF6 molecule as template
        d = sfbond_distance
        molecule = Atoms('S', positions=[(0.,0.,0.)]) 
        molecule += Atoms('6F', 
	                       positions=[(-d,0.,0.),
                                      (0.,-d,0.),
                                      (0.,0.,-d),
	                                  ( d,0.,0.),
                                      (0., d,0.),
                                      (0.,0., d)]
                     )
        sf6_constraints = self.constrainSF6(molecule)
        molecule.set_constraint(sf6_constraints)
        self._template_molecule = molecule
        

        # Create unit cell
        self._makeUnitCell()

        # Create supercell
        self._makeSuperCell()

    def _addWithConstraints(self,atoms,others):
        constraints = list()
        together = atoms + others
        constraints.extend(atoms.constraints)
        constraints.extend(others.constraints)
        together.set_constraint(constraints)
        return together
        

    def _makeUnitCell(self):
        lat = self._lat
        lat_half = self._lat / 2.

        # Set up unit cell with 1st SF6 molecule
        self._unit_cell = Atoms(cell=[lat,lat,lat], pbc=[1,1,1])
        self._unit_cell = self._addWithConstraints(self._unit_cell, self._template_molecule)

        # Add 2nd SF6 molecule
        molecule = self._template_molecule.copy()
        molecule.translate([lat_half,lat_half,lat_half])
        sf6_constraints = self.constrainSF6(molecule,shift_index=len(molecule))
        molecule.set_constraint(sf6_constraints)
        self._unit_cell = self._addWithConstraints(self._unit_cell, molecule)

    def getUnitCell(self):
        if not self._unit_cell:
            self._makeUnitCell()
        return self._unit_cell.copy()

    def _makeSuperCell(self):
        lat = self._lat
        sc  = self._supercell
        self.system = Atoms(cell=[sc[0]*lat,sc[1]*lat,sc[2]*lat], pbc=[1,1,1])
       
        ix = 0
        iy = 0
        iz = 0 
        for ix in range(sc[0]):
            for iy in range(sc[1]):
                for iz in range(sc[2]):
                    self._addUnitCell(ix,iy,iz)

    def _addUnitCell(self,ix,iy,iz):
        lat = self._lat
        vector = [ix*lat, iy*lat, iz*lat]
        molecules = self.getUnitCell()
        molecules.translate(vector)
        sf6_constraints = self.constrainSF6(molecules,shift_index=len(self.system))
        molecules.set_constraint(sf6_constraints)
        self.system = self._addWithConstraints(self.system, molecules) 

    # Constrain SF6 molecule - make rigid molecule
    def constrainSF6( self, molecule, shift_index=0 ):
        if isinstance(molecule.constraints,list):
            molecular_constraints = molecule.constraints
        elif molecule.constraints:
            molecular_constraints = [molecule.constraints]
        else:
            molecular_constraints = [] 

        # Get SF bond list
        s_index   = [x.index for x in molecule if x.symbol == 'S'][0]
        f_indices = [x.index for x in molecule if x.symbol == 'F']

        sf_bonds = list()
        for f_index in f_indices:
          sf_bond = [s_index+shift_index, f_index+shift_index]
          sf_bonds.append( [molecule.get_distance(s_index, f_index), sf_bond] )
          #sf_bonds.append(sf_bond)

        # Get FSF angle list
        fsf_angles = list()
        for f1 in f_indices:
          for f2 in f_indices:
            if f1 < f2:
              fsf_angle = [f1+shift_index, s_index+shift_index, f2+shift_index]
              fsf_angles.append( [molecule.get_angle([f1,s_index,f2]), fsf_angle] )


        molecular_constraints = FixInternals( bonds=sf_bonds, angles=fsf_angles, epsilon=1.e-4 )
        #molecular_constraints.append(FixInternals( bonds=sf_bonds,  epsilon=1.e-7 ))
        return molecular_constraints

    def lammpsParameters(self):
        # Simulation Parameters

        eps_ff = 70.60 # units: K
        sig_ff =  2.70 # units: Angstrom

        eps_ff *= units.kB / (units.kcal/units.mol) # units: kcal/mol

        print(eps_ff, sig_ff)

        # LAMMPS input
        header = ['units real', 
                  'atom_style molecular',
                  'bond_style harmonic',
                  'atom_modify map array sort 0 0']

        cmds = ["pair_style lj/cut 10.0", 
            "pair_coeff * * 0.0 0.0",
            "pair_coeff 2 2 "+str(eps_ff)+' '+str(sig_ff), 
            "bond_coeff * 100.0 1.565",
            "fix 1 all nve"]

        return header, cmds

