import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
import os
import time
import numpy as np
from mace.calculators import mace_off
import argparse
from mace.calculators import MACECalculator
import matplotlib.pyplot as plt
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
import warnings
import logging
from pathlib import Path
logging.getLogger("torch").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=FutureWarning)
import sys  


def parse_arguments():
    parser = argparse.ArgumentParser(description="Python code to run Langevin MD and use a heat reamp")
    parser.add_argument("input_file",type=str,help="Coordinate file that contains the initial configuration")
    parser.add_argument("output_file",type=str,help='XYZ file path and name to output properties')
    parser.add_argument("--start_temp",type=float,required=True,help='Starting temperature for the heat ramp')
    parser.add_argument("--final_temp",type=float,required=True,help='Final temperature for the heat ramp')
    parser.add_argument("--temp_steps",type=int,required=True,help="Number of steps needed to final temperature")
    parser.add_argument("--device",default='cpu',type=str,choices=['cpu','cuda'],help="Device on which to run the inference of the MLFF (cpu | cuda)")
    parser.add_argument("--timestep",type=float,default=0.1,help="Timestep for you MD simulation")
    parser.add_argument("--nstep",type=int,required=True,help="Number of steps in MD simulation")
    parser.add_argument("--model_path,",type=str,help='')
    parser.add_argument("--ensemble",type=str,default="npt",choices=['nve,nvt,npt'],help="Ensemble used for ensamble run (NVE | NPT | NVE)")
    parser.add_argument("--equilibration_steps",type=int,default=10000,help="Number of equilibration steps") 
    parser.add_argument("--npt_type",type=str,default="nose",choices=['nose','berendsen'],help="Type of NPT thermostat to use (nose | hoover)")
    return parser.parse_args()

def NPT_calc(init_conf, temp, calc, fname, s, T,timestep,npt_type):
    traj = f"{os.path.splitext(fname)[0]}.traj"
    log = f"{os.path.splitext(fname)[0]}.log"
    init_conf.calc = calc
    if npt_type == "nose":
        dyn = NPT(init_conf, timestep*units.fs, temperature_K=temp, trajectory=traj,logfile=log,externalstress=1.0*units.bar,ttime=20*units.fs,pfactor=2e6*units.fs**2,append_trajectory=True)

    if npt_type == "berendsen":
        dyn = Inhomogeneous_NPTBerendsen(init_conf, timestep*units.fs, temperature_K=temp, pressure_au = 1.01325 * units.bar , compressibility_au=4.57e-5 / units.bar,trajectory=traj,logfile=log,append_trajectory=True)
    time_fs = []
    temperature = []
    energies = []
    
    def write_frame(temp):
            dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
            dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
            dyn.atoms.write(fname, append=True)
            time_fs.append(dyn.get_time()/units.fs)
            time_fs.append(dyn.get_time()/units.fs)
            temperature.append(dyn.atoms.get_temperature())
            energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

    dyn.attach(lambda: write_frame(temp), interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))

def ramp_temperature(start_temp,end_temp,step,n_steps):
        temp = start_temp + (end_temp - start_temp) * step / n_steps
        return temp
        
def main():
    print('starting')
    args = parse_arguments()
    start_temp = args.start_temp  # Initial temperature in K
    end_temp = args.final_temp   # Final temperature in K
    n_steps = args.nstep   # Total number of MD steps
    device = str(args.device)
    timestep = args.timestep
    temp_steps = args.temp_steps
    input_file = args.input_file
    output_file = args.output_file
    model_path = None
    NPT_type = args.npt_type
    equilibration_steps = args.equilibration_steps

    print(f"Starting temperature at {start_temp}K ending in {end_temp}K in {temp_steps}. MD simualtion is run at {timestep} in {n_steps}")
    #if not model path is given use the MACE-off as default
    
    calc = mace_off(model="medium", device=device)
    print('Using MACE-off')

    #Read initial configuration
    init_conf = read(input_file)

    #Make a copy to not alter the original configuration
    current_conf = init_conf.copy()
    current_conf.calc = calc
    MaxwellBoltzmannDistribution(current_conf, temperature_K=start_temp) #Initialize temperature at starting temp
    Stationary(current_conf)
    ZeroRotation(current_conf)
    print('working')
    for step in range(temp_steps):
        # Update the target temperature for the ramp
        current_target_temp = ramp_temperature(start_temp,end_temp,step,temp_steps)
        fname = output_file
        print(f"Running MD calculation with {current_target_temp}K")
        
        # NVT Equilibration
        nvt = NVTBerendsen(
            atoms=current_conf,
            timestep=1.0 * units.fs,
            temperature_K=current_target_temp,
            taut=100.0 * units.fs,
            trajectory = f"{os.path.splitext(fname)[0]}_equilibration.traj",
            append_trajectory=True
        )
        nvt.run(equilibration_steps)


        # Run the MD simulation
        NPT_calc(current_conf, temp=current_target_temp, calc=calc, fname=fname, s=10, T=n_steps,timestep=timestep,npt_type=NPT_type)


if __name__ == '__main__':
    main()