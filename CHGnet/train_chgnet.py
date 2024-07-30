import pickle
import sys
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from datetime import datetime

import os 

pickle_file = sys.argv[1]
base_directory = sys.argv[2]

chgnet = CHGNet.load(check_cuda_mem = False)


def open_pickle(pickle_file):
    with open(pickle_file, "rb") as f:
        # Load the dictionary from the pickle file
        dataset_dict = pickle.load(f)
        return(dataset_dict)
    
def assign_variable(dataset_dict):
    structure = dataset_dict['structure']
    energies = dataset_dict['energies']
    forces = dataset_dict['forces']
    stresses = dataset_dict['stress']
    return(structure,energies,forces,stresses)

def load_dataset(structure,energies,forces,stresses):
    dataset= StructureData(
    structures=structure,
    energies=energies,
    forces=forces,
    stresses=stresses,  # can be None
    magmoms=None,  # can be None
    )
    return(dataset)

def freeze_layers():
    for layer in [
    chgnet.atom_embedding,
    chgnet.bond_embedding,
    chgnet.angle_embedding,
    chgnet.bond_basis_expansion,
    chgnet.angle_basis_expansion,
    chgnet.atom_conv_layers[:-1],
    chgnet.bond_conv_layers,
    chgnet.angle_layers,
]:
        for param in layer.parameters():
            param.requires_grad = False

def trainer_func():
        trainer = Trainer(
        model=chgnet,
        targets="efs",
        energy_loss_ratio=1,
        force_loss_ratio=1,
        stress_loss_ratio=1,
        optimizer="Adam",
        weight_decay=0,
        scheduler="CosLR",
        criterion="Huber",
        delta=0.1,
        epochs=30,
        starting_epoch=0,
        learning_rate=5e-3,
        use_device="cuda",
        check_cuda_mem = False,
        print_freq=100,
    )
        return(trainer)

def generate_unique_dir_path(base_name, base_directory):
    # Extract the base name without extension
    base = os.path.basename(base_name).split('.')[0]
    # Get the current date and time
    now = datetime.now()
    # Format the directory name with year, month, day, hour, minute, and second
    dir_name = now.strftime("%Y-%m-%d-%H-%M-%S")
    # Create the full path with base directory, base name, and formatted directory name
    full_path = os.path.join(base_directory, f"{base}-{dir_name}")
    return full_path


def main():
    dataset_dict=open_pickle(pickle_file)
    structure,energies,forces,stresses = assign_variable(dataset_dict)
    dataset = load_dataset(structure,energies,forces,stresses)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )
    freeze_layers()
    trainer = trainer_func()
    save_dir = generate_unique_dir_path(pickle_file,base_directory)
    print(f"Saving model to the directory {save_dir}")
    trainer.train(train_loader,val_loader,test_loader,save_dir=save_dir)


if __name__ == '__main__':
     main()


