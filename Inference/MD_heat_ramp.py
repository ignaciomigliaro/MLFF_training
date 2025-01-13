from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
import os
import time
import numpy as np
import pylab as pl
from IPython import display
np.random.seed(701) #just making sure the MD failure is reproducible
from mace.calculators import mace_off
import argparse
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
        
        


def parse_arguments():
    parser = argparse.ArgumentParser(description="Python code to run Langevin MD and use a heat reamp")
    parser.add_argument("input_file",type=str,help="Coordinate file that contains the initial configuration")
    parser.add_argument("output_file",type=str,require=True,help='XYZ file path and name to output properties')
    parser.add_argument("--start_temp",type=float,required=True,help='Starting temperature for the heat ramp')
    parser.add_argument("--final_temp",type=float,required=True,help='Final temperature for the heat ramp')
    parser.add_argument("--temp_steps",type=int,required=True,help="Number of steps needed to final temperature")
    parser.add_argument("--device",default='cpu',type=str,choices=['cpu','cuda'],help="Device on which to run the inference of the MLFF (cpu | cuda)")
    parser.add_argument("--timestep",type=float,default=0.1,help="Timestep for you MD simulation")
    parser.add_argument("--nstep",type=int,required=True,help="Number of steps in MD simulation")
    parser.add_argument("--model_path,",type=str,help='')
    parser.add_argument("--ensemble",type=str,default="npt",choices=['nve,nvt,npt'],help="Ensemble used for ensamble run (NVE | NPT | NVE)")
    return parser.parse_args()

def NPT_calc(init_conf, temp, calc, fname, s, T):
    init_conf.calc = calc
    dyn = NPT(init_conf, 1.0*units.fs, temperature_K=temp, trajectory='npt_600k_n50_T10ps.traj',logfile='npt.log',externalstress=1.0*units.bar,ttime=20*units.fs,pfactor=2e6*units.fs**2,append_trajectory=True)

    def write_frame(temp):
            dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
            dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
            dyn.atoms.write(fname, append=True)
            time_fs.append(dyn.get_time()/units.fs)
            temperature.append(dyn.atoms.get_temperature())
            energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

    dyn.attach(lambda: write_frame(current_target_temp), interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))


    def write_frame(temp):
        # Update energy and temperature data for plotting
        dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
        dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
        time_fs.append(dyn.get_time() / units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy() / len(dyn.atoms))


    dyn.attach(lambda: write_frame(current_target_temp), interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))

def ramp_temperature(start_temp,end_temp,step,n_steps):
        temp = start_temp + (end_temp - start_temp) * step / n_steps
        return temp

def main():
    args = parse_arguments()
    start_temp = args.start_temp  # Initial temperature in K
    end_temp = args.final_temp   # Final temperature in K
    n_steps = args.nsteps   # Total number of MD steps
    device = args.device
    temp_steps = args.temp_steps
    input_file = args.input_file
    output_file = args.output_file
    #if not model path is given use the MACE-off as default
    if args.model_path:
        calc = MACECalculator(args.model_path,device=device)
    else:
        calc = mace_off(model="medium", device=args.device)

    #Read initial configuration
    init_conf = read(input_file)

    #Make a copy to not alter the original configuration
    current_conf = init_conf.copy()
    current_conf.calc = calc
    MaxwellBoltzmannDistribution(current_conf, temperature_K=start_temp) #Initialize temperature at starting temp
    Stationary(current_conf)
    ZeroRotation(current_conf)

    for step in range(temp_steps):
        # Update the target temperature for the ramp
        current_target_temp = ramp_temperature(start_temp,end_temp,step,temp_steps)
        fname = output_file
        print(f"Running MD calculation with {t}K")
        
        # Run the MD simulation
        NPT_calc(current_conf, temp=current_target_temp, calc=calc, fname=fname, s=1, T=n_steps)
        