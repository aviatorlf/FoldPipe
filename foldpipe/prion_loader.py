import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from Bio import PDB
import glob

# Mapping of element symbols to their atomic numbers (Z)
ELEMENT_TO_Z = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
    # Add more as needed for specific PDBs
}

class PrionDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # We look for all .pdb files in the raw directory
        return glob.glob(os.path.join(self.raw_dir, '*.pdb'))
        
    @property
    def processed_file_names(self):
        return ['prion_data.pt']
        
    def download(self):
        # In a real scenario, you'd auto-download from PDB here if files are missing.
        # For now, we assume the user has placed .pdb files in the `data/prion/raw/` directory.
        pass
        
    def process(self):
        data_list = []
        parser = PDB.PDBParser(QUIET=True)
        
        raw_files = self.raw_file_names
        if not raw_files:
            print(f"No PDB files found in {self.raw_dir}. Please place .pdb files there.")
            return

        for raw_path in raw_files:
            print(f"Parsing {raw_path}...")
            structure = parser.get_structure("prion", raw_path)
            
            z_list = []
            pos_list = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            element = atom.element.strip().upper()
                            if element in ELEMENT_TO_Z:
                                z_list.append(ELEMENT_TO_Z[element])
                                pos_list.append(atom.coord)
                            else:
                                # Fallback for unknown elements
                                z_list.append(0)
                                pos_list.append(atom.coord)
                                
            z_tensor = torch.tensor(z_list, dtype=torch.long)
            pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
            
            # Create PyG Data object
            data = Data(z=z_tensor, pos=pos_tensor)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)
            
        self.save(data_list, self.processed_paths[0])
        print(f"Successfully processed {len(data_list)} PDB structures.")
