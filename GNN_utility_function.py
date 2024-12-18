from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, DataLoader as GeoDataLoader
import torch

def smiles_to_graph(smiles):
    """
    Convert SMILES string to RDKit Mol object.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return mol
    except Exception as e:
        print(f"Failed to convert SMILES to Mol: {smiles}, Error: {e}")
        return None

def mol_to_data(mol, target=None, additional_feature=None):
    """
    Convert RDKit Mol object to PyTorch Geometric Data object.
    """
    try:
        # Atom features
        x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).unsqueeze(1)

        # Edge indices
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]  # Undirected graph

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Target
        y = torch.tensor([target], dtype=torch.float) if target is not None else None

        # Additional features
        if additional_feature is not None:
            additional_feature = torch.tensor(additional_feature, dtype=torch.float).unsqueeze(0)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        if additional_feature is not None:
            data.additional_feature = additional_feature
        return data
    except Exception as e:
        print(f"Failed to process molecule: {mol}, Error: {e}")
        return None

def prepare_dataloader(df, target_column, batch_size=32, shuffle=True, generator=None):
    """
    Prepare PyTorch Geometric DataLoader from DataFrame.
    """
    data_list = []
    for _, row in df.iterrows():
        mol = row.get('graph', None)
        if isinstance(mol, str):
            mol = smiles_to_graph(mol)
        target = row[target_column] if target_column in row else None
        additional_feature = row.get('additional_feature', None)
        data = mol_to_data(mol, target, additional_feature)
        if data is not None:
            data_list.append(data)

    if not data_list:
        raise ValueError("No valid data points were created. Check your dataset and featurization.")

    loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=shuffle, generator=generator)
    return loader
