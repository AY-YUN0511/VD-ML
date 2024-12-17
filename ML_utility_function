from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp

def get_molecular_descriptors(mol):
    """
    Calculate molecular descriptors for a given molecule.
    """
    descriptors = {}
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except Exception as e:
            descriptors[name] = 0.0  # Assign default value on failure
            print(f"Descriptor calculation failed for {name}: {e}")
    return descriptors

def compute_and_scale_descriptors(df, scaler=None, fit_scaler=False, missing_val=0.0):
    """
    Compute molecular descriptors and scale them.
    """
    def process_smi(smi):
        mol = smiles_to_graph(smi)
        if mol is None:
            return None
        descriptors = get_molecular_descriptors(mol)
        descriptor_values = np.array(list(descriptors.values()), dtype=float)
        descriptor_values = np.nan_to_num(descriptor_values, nan=missing_val, posinf=missing_val, neginf=missing_val)
        return descriptor_values
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        descriptor_list = pool.map(process_smi, df['SMILES'])
    
    # Filtering
    valid_indices = [i for i, desc in enumerate(descriptor_list) if desc is not None]
    skipped_indices = [i for i, desc in enumerate(descriptor_list) if desc is None]
    descriptor_list = [desc for desc in descriptor_list if desc is not None]
    
    descriptor_names = list(Descriptors.descList[0][0] for _ in Descriptors.descList)
    additional_features = pd.DataFrame(descriptor_list, columns=descriptor_names)
    additional_features = additional_features.clip(lower=-1e6, upper=1e6)
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_descriptors = scaler.fit_transform(additional_features)
    else:
        scaled_descriptors = scaler.transform(additional_features)
    
    return {
        'scaled_descriptors': scaled_descriptors,
        'valid_indices': valid_indices,
        'skipped_indices': skipped_indices,
        'scaler': scaler,
        'descriptor_names': descriptor_names
    }

def compute_ecfp_fingerprints(df):
    """
    Compute Extended-Connectivity Fingerprints (ECFP) for molecules in the DataFrame.
    """
    def process_smi(smi):
        mol = smiles_to_graph(smi)
        if mol is None:
            return None
        ecfp_bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ecfp_array = np.zeros((2048,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(ecfp_bitvect, ecfp_array)
        return ecfp_array
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        ecfp_list = pool.map(process_smi, df['SMILES'])
    
    # Filtering
    valid_indices = [i for i, ecfp in enumerate(ecfp_list) if ecfp is not None]
    ecfp_features = np.array([ecfp for ecfp in ecfp_list if ecfp is not None])
    
    return {
        'ecfp_features': ecfp_features,
        'valid_indices': valid_indices
    }
