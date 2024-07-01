from mace.calculators import mace_mp
from ase import build
from ase.io import read,write
import numpy as np
from pymatgen.core import Structure
from pathlib import Path 
from chgnet.model import CHGNet
from ase import Atom, Atoms
import os 
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import StructOptimizer
import crystal_toolkit.components as ctc
import plotly.graph_objects as go
from crystal_toolkit.settings import SETTINGS
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from pymatgen.core import Structure
import argparse
from tqdm import tqdm
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
from dash import Dash, dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from pymatgen.core.structure import Structure
import crystal_toolkit.components as ctc
from crystal_toolkit.settings import SETTINGS
import warnings 
import seaborn as sns
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import VoronoiNN
from mace.calculators import MACECalculator
import torch
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats


#Parse DFT data
def read_dft(filepath,mlff_opt=False):
    """This function reads the DFT files from the given Directory, groups the files by the number of Atoms."""
    atoms_list = []
    opt_atoms_list = []
    n = 0
    total_iterations = len(os.listdir(filepath))
    with tqdm(total=total_iterations, desc='Processing') as pbar:
        for i in os.listdir(filepath):
            if os.path.isdir(filepath+i):
                if mlff_opt == True:
                    tote = filepath + i + '/log.tote'
                    mlff_opt_energy = parse_tote(tote)
                OUTCAR = filepath + i + '/OUTCAR'
                try:
                    single_file_atom = read(OUTCAR, format='vasp-out', index=':')
                    last_energy = single_file_atom[-1]
                    atom = single_file_atom[0]
                    atom.info['file'] = filepath + i
                    last_energy.info['file'] = filepath + i 
                    atom.info['step'] = 'first step'
                    last_energy.info['step'] = 'last step'
                    if mlff_opt is True:
                        atom.info['mlff_opt_energy'] = mlff_opt_energy
                    if atom is not None and last_energy is not None:
                        atoms_list.append(atom)
                        opt_atoms_list.append(last_energy)     
                except Exception as e:
                    print(f"Error reading file: {OUTCAR}")
                    print(f"Error details: {e}")
                    continue
                finally:
                    pbar.update(1)
    return(atoms_list,opt_atoms_list)

def parse_tote(work_path):
    try:
        with open(work_path) as f:
            lines = f.readlines()
        energy = float(lines[-1].split()[0])
    except Exception as e: 
         energy = np.nan
         print(e)
    return energy   
    
def create_empty_atom(atom):
    empty_atom = Atoms(numbers=atom.get_atomic_numbers(), positions=atom.get_positions(),cell=atom.get_cell())
    return empty_atom
#Create a list of atoms objects with no energy (needed for running inference with MLFF)
def create_empty_atom_list(atoms_list):
    no_energy_atoms = []
    for atom in atoms_list: 
        empty_atom = create_empty_atom(atom)
        no_energy_atoms.append(empty_atom)
    return no_energy_atoms
#Run inference using MACE MLFF
def mace_inference(atoms_list,model_path=None):
    no_energy_atoms = create_empty_atom_list(atoms_list)
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            if model_path:
                print('Using model to calculate')
                calc = MACECalculator(model_paths=model_path,device='cpu')
            else: 
                calc = mace_mp(model="large", dispersion=True, device='cpu', verbose=True)
    for atom, atom_ne in zip(atoms_list, no_energy_atoms):
        atom_ne.calc = calc
        e = atom_ne.get_total_energy()
        f = atom_ne.get_forces()
        atom.info['mace_energy'] = e
        atom.info['mace_forces'] = f
    return atoms_list
#Run inference using Chgnet MLFF
def chgnet_inference(atoms_list, model_path=None):
    no_energy_atoms = create_empty_atom_list(atoms_list)
    for atom, atom_ne in zip(atoms_list, no_energy_atoms):
        structure = AseAtomsAdaptor().get_structure(atom_ne)
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                if model_path:
                    loaded_model = CHGNet.from_file(model_path, use_device='cpu', verbose=False)
                    loaded_model = loaded_model.to(torch.float32)
                    prediction = loaded_model.predict_structure(structure)
                else:
                    chgnet = CHGNet.load()
                    prediction = chgnet.predict_structure(structure)
        
        total = sum(len(site.species) for site in structure.sites)
        energy = prediction['e'] * total
        forces = prediction['f'] 
        atom.info['chgnet_energy'] = energy
        atom.info['chgnet_forces'] = np.array(forces)
    return atoms_list
#Calculates the energy differences from starting to finished structure
def opt_energy_diff(atoms_list,opt_atoms_list):
    "This function is to calculate the energy difference between starting energy and final from optimization"
    for atom, opt_atom in zip(atoms_list,opt_atoms_list):
            e_diff= atom.get_total_energy() - opt_atom.get_total_energy()
            atom.info['opt_e_diff'] = e_diff

            if atom.info.get('chgnet_energy') is not None and opt_atom.info.get('chgnet_energy') is not None:
                e_diff_chgnet= atom.info.get('chgnet_energy') - opt_atom.info .get('chgnet_energy')
                atom.info['opt_e_diff_chgnet'] = e_diff_chgnet

            if atom.info.get('mace_energy') is not None and opt_atom.info.get('mace_energy') is not None:
                e_diff_mace= atom.info.get('mace_energy') - opt_atom.info .get('mace_energy')
                atom.info['opt_e_diff_mace'] = e_diff_mace
    return(atoms_list)
#Groups atoms if they have all the same atomic symbols
def group_atoms_by_number_if_same_symbols(atoms_list,opt_atoms_list):
    if not atoms_list :
        raise ValueError("The input lists are empty")
    
    # Check if all atoms objects have the same atomic symbols
    reference_symbols = set(atoms_list[0].get_chemical_symbols())
    for atoms, opt_atoms in zip(atoms_list, opt_atoms_list):
        if set(atoms.get_chemical_symbols()) != reference_symbols or \
                set(opt_atoms.get_chemical_symbols()) != reference_symbols:
            raise ValueError("Not all atoms objects have the same atomic symbols")
    
    # Create dictionaries to hold lists of Atoms objects grouped by number of atoms
    grouped_atoms = {}
    
    # Populate the dictionaries
    for atoms in atoms_list:
        num_atoms = len(atoms)
        if num_atoms not in grouped_atoms:
            grouped_atoms[num_atoms] = []
        grouped_atoms[num_atoms].append(atoms)

    
    return grouped_atoms
#Ranks the atoms by their energy
def rank_atoms_by_energy(grouped_atoms):
    """Sorts groups of atoms based on the total number of atoms and then sorts the atoms within each group by their total energy."""
    # Sort groups based on the total number of atoms
    grouped_atoms_sorted = dict(sorted(grouped_atoms.items(), key=lambda item: item[0]))
    
    # Sort atoms within each group by their total energy
    for num_atoms, atoms_list in grouped_atoms_sorted.items():
        grouped_atoms_sorted[num_atoms] = sorted(atoms_list, key=lambda atom: atom.get_total_energy())
    
    return grouped_atoms_sorted
#Prints the energies if needed 
def print_sorted_energies(grouped_atoms_sorted,mace_flag=None):
    """Prints the total energy and MACE energy of each atom in each group in the sorted dictionary."""
    for num_atoms, atoms_list in grouped_atoms_sorted.items():
        print(f"Total energy for group with {num_atoms} atoms:")
        for atom in atoms_list:
            file_path = atom.info.get('file', 'File path not available')
            total_energy = atom.get_total_energy()
            chgnet_energy = atom.info.get('chgnet_inference','Chgnet energy not available')
            basename = os.path.basename(file_path)
            if mace_flag:
                mace_energy = atom.info.get('mace_energy', 'MACE energy not available')
                print(f"File: {basename}, Total Energy: {total_energy}, CHGnet Energy: {chgnet_energy} MACE Energy: {mace_energy}")
            else: 
                print(f"File: {basename}, Total Energy: {total_energy}, CHGnet Energy: {chgnet_energy} ")
#calculates MSE of forces
def mean_square_error(true_forces, predicted_forces):
    return np.mean((true_forces - predicted_forces) ** 2)

#Gets the sorted atoms, and creates a Dataframe with multiple properties.
def get_sorted_energies_dataframe(grouped_atoms_sorted, mace_flag=None,mlff_opt = None,calc_correlation=False):
    """Returns a DataFrame with the total energy, CHGnet energy, MACE energy (optional), their differences, and the MSE of the forces for each atom in each group in the sorted dictionary."""
    data = []

    for num_atoms, atoms_list in grouped_atoms_sorted.items():
        for atom in atoms_list:
            file_path = atom.info.get('file', 'File path not available')
            total_energy = atom.get_total_energy()
            chgnet_energy = atom.info.get('chgnet_energy', 'Chgnet energy not available')
            dft_diff = atom.info.get('opt_e_diff')
            basename = os.path.basename(file_path)
            if mlff_opt:
                mlff_opt_energy = atom.info['mlff_opt_energy']
            # Calculate the differences
            chgnet_delta_E = None
            mace_delta_E = None
            chgnet_opt_delta = None
            mace_opt_delta = None
            chgnet_mse = None
            mace_mse = None

            if isinstance(chgnet_energy, (int, float)):
                chgnet_delta_E = total_energy - chgnet_energy
                chgnet_opt_diff = atom.info.get('opt_e_diff_chgnet', 'CHGnet opt energy not available')
                chgnet_opt_delta = dft_diff - chgnet_opt_diff
                
                # Calculate CHGnet force MSE
                chgnet_forces = atom.info.get('chgnet_forces')
                if chgnet_forces is not None:
                    dft_forces = atom.get_forces()
                    chgnet_mse = mean_square_error(dft_forces, chgnet_forces)

            if mace_flag:
                mace_energy = atom.info.get('mace_energy', 'MACE energy not available')
                mace_opt_diff = atom.info.get('opt_e_diff_mace', 'MACE opt energy not available')
                mace_opt_delta = dft_diff - mace_opt_diff

                if isinstance(mace_energy, (int, float)):
                    mace_delta_E = mace_energy - total_energy
                
                # Calculate MACE force MSE
                mace_forces = atom.info.get('mace_forces')
                if mace_forces is not None:
                    dft_forces = atom.get_forces()
                    mace_mse = mean_square_error(dft_forces, mace_forces)

                data.append({
                    'File': basename,
                    'MLFF_opt_E':mlff_opt_energy,+ 
                    'DFT E': round(total_energy/num_atoms, 3),
                    'Opt Δ (DFT)': round(dft_diff/num_atoms, 3),
                    'MLFF E': round(mace_energy/num_atoms, 3),
                    'ΔE': round(mace_delta_E/num_atoms, 3),
                    'Opt ΔE (MLFF)': round(mace_opt_diff/num_atoms, 3),
                    'Opt ΔΔE': round(mace_opt_delta/num_atoms, 3),
                    'Forces MSE': round(mace_mse, 3) if mace_mse is not None else None,
                    'natom': num_atoms,
                    
                })
            else:
                data.append({
                    'File': basename,
                    'DFT E': round(total_energy/num_atoms, 3),
                    'Opt Δ (DFT)': round(dft_diff/num_atoms, 3),
                    'MLFF E': round(chgnet_energy/num_atoms, 3),
                    'ΔE': round(chgnet_delta_E/num_atoms, 3),
                    'Opt ΔE (MLFF)': round(chgnet_opt_diff/num_atoms, 3),
                    'Opt ΔΔE': round(chgnet_opt_delta/num_atoms, 3),
                    'Forces MSE': round(chgnet_mse, 3) if chgnet_mse is not None else None,
                    'natom': num_atoms
                })

    df = pd.DataFrame(data)
    
    # Calculate Spearman rank correlation for each group of `natom`
    if calc_correlation:
        # Calculate Spearman rank correlation for all structures
        df['DFT Rank'] = df['DFT E'].rank(ascending=True, method='dense').astype(int)
        df['MLFF Rank'] = df['MLFF E'].rank(ascending=True, method='dense',numeric_only=True).astype(int)
        correlation, _ = pearsonr(df['DFT Rank'], df['MLFF Rank'])
        df['Rank ρ'] = correlation
    else:
        df['Rank ρ'] = None

    return df

#Calls all inference functions and calculates energies for all of the test set
def inference(atoms_list, opt_atoms_list, model_path, mace_flag=None, mlff_opt=None, calc_correlation=False):
    if mace_flag:
        print('Running MACE')
        atoms_list = mace_inference(atoms_list,model_path)
        opt_atoms_list = mace_inference(opt_atoms_list,model_path)
    else:
        print('Running CHgnet')
        #Run CHGNet inference
        atoms_list = chgnet_inference(atoms_list,model_path)
        opt_atoms_list = chgnet_inference(opt_atoms_list,model_path)
    #Find difference in atoms
    atoms_list = opt_energy_diff(atoms_list,opt_atoms_list)
    #Group and sort atoms by natoms
    grouped_atoms = group_atoms_by_number_if_same_symbols(atoms_list,opt_atoms_list)
    grouped_atoms_sorted = rank_atoms_by_energy(grouped_atoms)
    #Create dataframe
    df=get_sorted_energies_dataframe(grouped_atoms_sorted,mace_flag,mlff_opt=mlff_opt,calc_correlation=calc_correlation)
    return(df)
#Plots the mean absolute error of the energies for each model
def plot_mae_comparison(dataframes, dataframe_names, mace_flag=None):
    chgnet_MAE = []
    chgnet_opt_MAE = []
    
    # Calculate mean absolute errors (MAEs) for each dataframe
    for df in dataframes: 
        chgnet_error = df['ΔE'].abs().mean()
        chgnet_MAE.append(chgnet_error)
        
        chgnet_opt_error = df['Opt ΔΔE'].abs().mean()
        chgnet_opt_MAE.append(chgnet_opt_error)
    
    # Create the bar graph
    x = range(len(dataframes))  # X-axis positions for each dataframe
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()
    
    # Plot CHGnet MAEs
    ax.bar(x, chgnet_MAE, width, label='CHGnet MAE')
    
    # Plot optimized CHGnet MAEs with an offset to avoid overlapping
    ax.bar([p + width for p in x], chgnet_opt_MAE, width, label='Optimized CHGnet MAE')

    # Add labels, title, and legend
    ax.set_xlabel('Models')
    ax.set_ylabel('Mean Absolute Error (eV)')
    ax.set_title('Comparison of Mean Absolute Errors')
    ax.set_xticks([p + width/2 for p in x])
    ax.set_xticklabels(dataframe_names,rotation=90)
    ax.legend()
    plt.show()
#Creates bar graph of MSE of Forces of each model
def plot_mse_comparison(dataframes, dataframe_names, mace_flag=None):
    chgnet_MSE = []
    
    # Calculate mean squared errors (MSEs) for each dataframe
    for df in dataframes: 
        chgnet_mse = df['Forces MSE'].mean()
        chgnet_MSE.append(chgnet_mse)
    
    # Create the bar graph
    x = range(len(dataframes))  # X-axis positions for each dataframe
    width = 0.55  # Width of the bars

    fig, ax = plt.subplots()
    
    # Plot CHGnet MSEs
    ax.bar(x, chgnet_MSE, width, label='CHGnet Force MSE')

    # Add labels, title, and legend
    ax.set_xlabel('Models')
    ax.set_ylabel('Mean Squared Error (eV/Å^2)')
    ax.set_title('Comparison of Mean Squared Errors for Forces')
    ax.set_xticks(x)
    ax.set_xticklabels(dataframe_names, rotation=90)
    ax.legend()

    plt.tight_layout()  # Adjust layout to make room for the rotated labels
    plt.show()    
#Createas a bar graph for the Rank correlation of each of the models
def plot_rank_correlation(data_dicts,dataframe_names):
    dataframes = []
    for data_dict in data_dicts:
        df = pd.DataFrame(data_dict)
        dataframes.append(df)
    
    natom_values = sorted({natom for df in dataframes for natom in df['natom'].unique()})
    num_dataframes = len(dataframes)
    num_natoms = len(natom_values)
    
    # Calculate the width for each group of bars and the total figure width
    bar_width = 0.8 / num_natoms
    x_positions = np.arange(num_dataframes)  # X positions for dataframes

    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, natom in enumerate(natom_values):
        rank_rho_per_dataframe = {}
        for df_idx, df in enumerate(dataframes, start=1):
            subset_df = df[df['natom'] == natom]
            unique_rank_rho_values = subset_df['Rank ρ'].unique()
            
            if len(unique_rank_rho_values) == 1:
                rank_rho_per_dataframe[df_idx] = unique_rank_rho_values[0]
            else:
                print(f"Warning: Inconsistent 'Rank ρ' values for natom{natom} in dataframe {df_idx}.")
        
        ranks = [rank_rho_per_dataframe.get(i, 0) for i in range(1, num_dataframes + 1)]
        ax.bar(x_positions + (idx - num_natoms/2) * bar_width, ranks, width=bar_width, label=f'natom{natom}')

    ax.set_xlabel('Dataframe Name')
    ax.set_ylabel('Rank ρ')
    ax.set_title('Rank ρ for Different natom Values Across Dataframes')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataframe_names, rotation=90)  # Rotate x-axis labels to 90 degrees
    ax.legend(title='natom', bbox_to_anchor=(1.02, 1), loc='upper left')  # Move legend outside of the plot area
    
    ax.set_ylim(0, 1)  # Set y-axis limits
    
    plt.tight_layout()
    plt.show()
#Optimizes the initial structure of an outcar file and compares to the optimized structure by DFT for Chgnet
def plot_mlff_dft_rank(df):
    DFT_Rank = df['DFT Rank']
    MLFF_Rank = df['MLFF Rank']

    # Create scatter plot
    plt.scatter(DFT_Rank, MLFF_Rank, label='Data points',color='black')

    # Fit a line to the data using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(DFT_Rank, MLFF_Rank)
    trendline = slope * DFT_Rank + intercept

    # Calculate Pearson correlation coefficient
    pearson_corr, _ = stats.pearsonr(DFT_Rank, MLFF_Rank)

    # Plot the trendline
    plt.plot(DFT_Rank, trendline, color='red', label='Trendline')

    # Calculate and display the R^2 value
    r_squared = r_value**2
    plt.text(5, max(MLFF_Rank/2), f'$R^2 = {r_squared:.3f}$', color='black')

    # Display Pearson correlation coefficient

    # Add labels and legend
    plt.xlabel('DFT Rank')
    plt.ylabel('MLFF Rank')
    plt.title('Rank comparison of MLFF and DFT')
    plt.legend()

    # Show plot
    plt.show()
def optimize(model, outcar_path, verbose=False,fmax=0.05):
    loaded_model = CHGNet.from_file(model, use_device='cpu', verbose=verbose)
    loaded_model = loaded_model.to(torch.float32)
    all_atoms = read(outcar_path, index=':')
    first_atom = all_atoms[0]
    last_atom = all_atoms[-1]
    dft_energy = last_atom.get_total_energy()
    first_struct = AseAtomsAdaptor().get_structure(first_atom)
    trajectory = StructOptimizer(loaded_model).relax(first_struct, verbose=verbose, relax_cell=True, fmax=fmax)["trajectory"]
    last_struct = AseAtomsAdaptor().get_structure(last_atom)
    return (trajectory, first_struct, last_struct,dft_energy)
#Needed to run the dash app 
def prepare_data(trajectory):
    """Prepare the input data from the trajectory."""
    e_col = "Energy (eV)"
    force_col = "Force (eV/Å)"
    
    # Create DataFrame for energy and force
    df_traj = pd.DataFrame(trajectory.energies, columns=[e_col])
    df_traj[force_col] = [
        np.linalg.norm(force, axis=1).mean()  # mean of norm of force on each atom
        for force in trajectory.forces
    ]
    df_traj.index.name = "step"
    
    return df_traj, e_col, force_col
#Dash app to show the optimization
def create_dash_app(trajectory, df_traj, e_col, force_col, structure, dft_energy):
    """Create and run the Dash app for visualizing the structure relaxation trajectory."""
    mp_id = 'NbOC'
    app = Dash(prevent_initial_callbacks=True, assets_folder=SETTINGS.ASSETS_PATH)

    step_size = max(1, len(trajectory) // 20)  # ensure slider has max 20 steps
    slider = dcc.Slider(
        id="slider", min=0, max=len(trajectory) - 1, step=step_size, updatemode="drag"
    )

    def plot_energy_and_forces(df: pd.DataFrame, step: int, e_col: str, force_col: str, title: str) -> go.Figure:
        """Plot energy and forces as a function of relaxation step."""
        fig = go.Figure()
        # energy trace = primary y-axis
        fig.add_trace(go.Scatter(x=df.index, y=df[e_col], mode="lines", name="Energy"))
        # get energy line color
        line_color = fig.data[0].line.color

        # forces trace = secondary y-axis
        fig.add_trace(
            go.Scatter(x=df.index, y=df[force_col], mode="lines", name="Forces", yaxis="y2")
        )

        fig.update_layout(
            template="plotly_white",
            title=title,
            xaxis=dict(title="Relaxation Step"),
            yaxis=dict(title=e_col),
            yaxis2=dict(title=force_col, overlaying="y", side="right"),
            legend=dict(yanchor="top", y=1, xanchor="right", x=1),
        )

        # vertical line at the specified step
        fig.add_vline(x=step, line=dict(dash="dash", width=1))

        # horizontal line for DFT final energy
        anno = dict(text="DFT final energy", yanchor="top")
        fig.add_hline(
            y=dft_energy,
            line=dict(dash="dot", width=1, color=line_color),
            annotation=anno,
        )

        return fig

    def make_title(spg_symbol: str, spg_num: int) -> str:
        """Return a title for the figure."""
        href = f"https://materialsproject.org/materials/{mp_id}/"
        return f"<a {href=}>{mp_id}</a> - {spg_symbol} ({spg_num})"

    title = make_title(*structure.get_space_group_info())

    graph = dcc.Graph(
        id="fig",
        figure=plot_energy_and_forces(df_traj, 0, e_col, force_col, title),
        style={"maxWidth": "50%"},
    )

    struct_comp = ctc.StructureMoleculeComponent(id="structure", struct_or_mol=structure)

    app.layout = html.Div(
        [
            html.H1(
                "Structure Relaxation Trajectory", style=dict(margin="1em", fontSize="2em")
            ),
            html.P("Drag slider to see structure at different relaxation steps."),
            slider,
            html.Div([struct_comp.layout(), graph], style=dict(display="flex", gap="2em")),
        ],
        style=dict(margin="auto", textAlign="center", maxWidth="1200px", padding="2em"),
    )

    ctc.register_crystal_toolkit(app=app, layout=app.layout)

    @app.callback(
        Output(struct_comp.id(), "data"), Output(graph, "figure"), Input(slider, "value")
    )
    def update_structure(step: int) -> tuple[Structure, go.Figure]:
        """Update the structure displayed in the StructureMoleculeComponent and the
        dashed vertical line in the figure when the slider is moved.
        """
        lattice = trajectory.cells[step]
        coords = trajectory.atom_positions[step]
        structure.lattice = lattice  # update structure in place for efficiency
        assert len(structure) == len(coords)
        for site, coord in zip(structure, coords):
            site.coords = coord

        title = make_title(*structure.get_space_group_info())
        fig = plot_energy_and_forces(df_traj, step, e_col, force_col, title)

        return structure, fig

    app.run_server(debug=True, use_reloader=False)
# Function to integrate data preparation and app creation
def visualize_trajectory(trajectory, structure, dft_energy):
    df_traj, e_col, force_col = prepare_data(trajectory)
    create_dash_app(trajectory, df_traj, e_col, force_col, structure, dft_energy)
#Calculates bond distance of structure object
def calculate_bond_distances(structure):
    # Initialize the Voronoi nearest neighbors finder
    voro_nn = VoronoiNN()
    bond_distances = []
    for i in range(len(structure)):
        nn_info = voro_nn.get_nn_info(structure, i)
        for neighbor_info in nn_info:
            neighbor_site = neighbor_info['site']
            # Check if the neighbor site exists in the structure
            if neighbor_site in structure:
                bond_distances.append(structure[i].distance(neighbor_site))
    return bond_distances
#Calculates bond angles of Structure object
def calculate_bond_angles(structure):
    # Initialize the Voronoi nearest neighbors finder
    voro_nn = VoronoiNN()
    bond_angles = []
    for i in range(len(structure)):
        nn_info = voro_nn.get_nn_info(structure, i)
        neighbor_indices = []
        for neighbor_info in nn_info:
            neighbor_site = neighbor_info['site']
            # Check if the neighbor site exists in the structure
            if neighbor_site in structure:
                neighbor_indices.append(structure.index(neighbor_site))
        for j in range(len(neighbor_indices)):
            for k in range(j + 1, len(neighbor_indices)):
                angle = structure.get_angle(i, neighbor_indices[j], neighbor_indices[k])
                bond_angles.append(angle)
    return bond_angles
#Plots KDE plots of bond length 
def plot_kde(bond_data, title, xlabel, color, label):
    sns.kdeplot(bond_data, fill=True, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
#Plots KDE of bond angles. 
def plot_bond_distributions(structure1, structure2):
    # Calculate bond distances and bond angles
    bond_distances_1 = calculate_bond_distances(structure1)
    bond_distances_2 = calculate_bond_distances(structure2)

    bond_angles_1 = calculate_bond_angles(structure1)
    bond_angles_2 = calculate_bond_angles(structure2)

    # Plot bond distance distributions
    plt.subplot(1, 2, 1)
    plot_kde(bond_distances_1, "Bond Distance Distribution", "Bond Distance (Å)", "blue", "MLFF optimized")
    plot_kde(bond_distances_2, "Bond Distance Distribution", "Bond Distance (Å)", "red", "DFT optimized")
    plt.legend(fontsize=6, loc='upper left')

    # Plot bond angle distributions
    plt.subplot(1, 2, 2)
    plot_kde(bond_angles_1, "Bond Angle Distribution", "Bond Angle (degrees)", "green", "MLFF optimized")
    plot_kde(bond_angles_2, "Bond Angle Distribution", "Bond Angle (degrees)", "orange", "DFT optimized")
    plt.legend(fontsize=6)

    plt.tight_layout()
    plt.show()
#Get the lattice parameters of MLFF and DFT optimized structure
def get_lattice_params(struct):
    return struct.lattice.a, struct.lattice.b, struct.lattice.c
#Prints the crystal lattice parameters
def optimization_summary(mlff_struct, dft_struct):
    mlff_volume = round(mlff_struct.volume, 3)
    dft_volume = round(dft_struct.volume, 3)
    mlff_distance = round(mlff_struct.get_distance(0, 1), 3)
    dft_distance = round(dft_struct.get_distance(0, 1), 3)

    mlff_a, mlff_b, mlff_c = get_lattice_params(mlff_struct)
    dft_a, dft_b, dft_c = get_lattice_params(dft_struct)

    # Rounding to 3 decimal places
    mlff_a = round(mlff_a, 3)
    mlff_b = round(mlff_b, 3)
    mlff_c = round(mlff_c, 3)
    dft_a = round(dft_a, 3)
    dft_b = round(dft_b, 3)
    dft_c = round(dft_c, 3)

    # Calculating differences (dft - mlff)
    volume_diff = round(dft_volume - mlff_volume, 3)
    distance_diff = round(dft_distance - mlff_distance, 3)
    a_diff = round(dft_a - mlff_a, 3)
    b_diff = round(dft_b - mlff_b, 3)
    c_diff = round(dft_c - mlff_c, 3)

    # Printing the comparison and differences
    print("MLFF vs DFT comparison:")
    print(f"Volume: MLFF = {mlff_volume}, DFT = {dft_volume}, Difference (DFT - MLFF) = {volume_diff}")
    print(f"Distance between atoms 0 and 1: MLFF = {mlff_distance}, DFT = {dft_distance}, Difference (DFT - MLFF) = {distance_diff}")
    print(f"Lattice parameters a: MLFF = {mlff_a}, DFT = {dft_a}, Difference (DFT - MLFF) = {a_diff}")
    print(f"Lattice parameters b: MLFF = {mlff_b}, DFT = {dft_b}, Difference (DFT - MLFF) = {b_diff}")
    print(f"Lattice parameters c: MLFF = {mlff_c}, DFT = {dft_c}, Difference (DFT - MLFF) = {c_diff}")
#Optimizes structure with MACE
def mace_optimize(model_path,outcar_path,verbose=False,fmax=0.005):
    #Function to optimize structure using MACE model
    from mace.calculators import MACECalculator
    from ase.optimize import FIRE
    all_atoms = read(outcar_path, index=':')
    first_atom = all_atoms[0]
    last_atom = all_atoms[-1]
    calculator = MACECalculator(model_path,device='cpu')
    first_atom.calc = calculator
    dyn = FIRE(first_atom)
    dyn.run(fmax=fmax)
    opt_struct = AseAtomsAdaptor().get_structure(first_atom)
    dft_struct = AseAtomsAdaptor().get_structure(last_atom)
    return(opt_struct,dft_struct)

def load_and_predict(model_path, structure):
    model = CHGNet.from_file(model_path, use_device="cpu", verbose=False)
    prediction = model.predict_structure(structure)
    total = sum(len(site.species) for site in structure.sites)
    return prediction['e'] * total

def energy_variance(filepath, model1, model2, model3):
    structure = Structure.from_file(filepath)
    with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull):
                energies = [
                    load_and_predict(model1, structure),
                    load_and_predict(model2, structure),
                    load_and_predict(model3, structure)
                ]
    return energies,np.std(energies)