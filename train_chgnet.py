import pickle
import sys
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader

pickle_file = sys.argv[1]
chgnet = CHGNet.load()


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
        stress_loss_ratio=0.1,
        optimizer="Adam",
        weight_decay=0,
        scheduler="CosLR",
        criterion="Huber",
        delta=0.1,
        epochs=50,
        starting_epoch=0,
        learning_rate=1e-3,
        use_device="cpu",
        print_freq=100,
    )
        return(trainer)

def main():
    dataset_dict=open_pickle(pickle_file)
    structure,energies,forces,stresses = assign_variable(dataset_dict)
    dataset = load_dataset(structure,energies,forces,stresses)
    train_loader, val_loader, test_loader = get_train_val_test_loader(
    dataset, batch_size=16, train_ratio=0.9, val_ratio=0.05
    )
    freeze_layers()
    trainer = trainer_func()
    trainer.train(train_loader,val_loader,test_loader)


if __name__ == '__main__':
     main()


