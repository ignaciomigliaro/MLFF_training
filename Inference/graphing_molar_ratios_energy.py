import os
import sys
import matplotlib.pyplot as plt

def parse_energy_values(base_dir):
    energies = []
    total_atoms = 64
    max_dopants = 32  # since we have 32 O atoms, the maximum number of C dopants is 32
    
    for x in range(max_dopants + 1):
        dir_name = f'structure_{x}_dopes_1'
        file_path = os.path.join(base_dir, dir_name, 'log.tote')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                energy = float(file.readline().strip())
                energies.append(energy / total_atoms)  # Normalize energy per atom
        else:
            energies.append(None)  # Handle cases where the file might be missing
    
    return energies

def plot_energies(energies):
    x_values = [x / 32 for x in range(len(energies))]  # Convert to molar fraction
    y_values = [e for e in energies if e is not None]
    
    # Set pure NbO energy (x=0) as reference
    nb_o_energy = y_values[0] if y_values else 0
    y_values_relative = [e - nb_o_energy for e in y_values]
    
    valid_x_values = [x_values[i] for i in range(len(energies)) if energies[i] is not None]

    plt.figure(figsize=(10, 6))
    plt.plot(valid_x_values, y_values_relative, marker='o')
    plt.title('Relative Normalized Energy vs. Molar Fraction of Carbon Atoms Doped')
    plt.xlabel('Molar Fraction of Carbon Atoms Doped')
    plt.ylabel('Relative Energy per Atom (eV)')
    
    # Set x-ticks with custom labels
    tick_positions = [i / 32 for i in range(33)]
    tick_labels = [f'{round(i / 32, 2):.2f}' for i in range(33)]
    tick_labels[0] = 'NbO'  # Label for x = 0
    tick_labels[-1] = 'NbC'  # Label for x = 1
    
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=90)  # Rotate labels to avoid overlap
    plt.grid(False)  # Remove gridlines
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/your/data")
        sys.exit(1)

    base_dir = sys.argv[1]

    energies = parse_energy_values(base_dir)
    plot_energies(energies)
