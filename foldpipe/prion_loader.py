import os
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from Bio import PDB
import glob

ELEMENT_TO_Z = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
}

class PrionStreamer(IterableDataset):
    """
    True O(1) streaming parser for raw PDB files. 
    Does NOT use PyG InMemoryDataset to avoid building a data_list.
    """
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.raw_files = glob.glob(os.path.join(self.raw_dir, '*.pdb'))
        
    def __iter__(self):
        if not self.raw_files:
            print(f"No PDB files found in {self.raw_dir}.")
            return

        parser = PDB.PDBParser(QUIET=True)
        
        for raw_path in self.raw_files:
            structure = parser.get_structure("prion", raw_path)
            z_list = []
            pos_list = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            element = atom.element.strip().upper()
                            z_list.append(ELEMENT_TO_Z.get(element, 0))
                            pos_list.append(atom.coord)
                                
            z_tensor = torch.tensor(z_list, dtype=torch.long)
            pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
            
            yield Data(z=z_tensor, pos=pos_tensor)
