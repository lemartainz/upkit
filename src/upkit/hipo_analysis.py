# %%
import uproot
import numpy as np
import pandas as pd
import vector as vec
import awkward as ak
import sys

### Analysis Class ###
class RootAnalysis:
    """
        Reads ROOT files using uproot, applies cuts and saves modified data. 

        Attributes
        ===========
        file_name: str
            The name of the ROOT file to analyze.
        tree_name: str
            The name of the tree within the ROOT file.
        data: dict
            A dictionary to hold the loaded data from the ROOT file.
        file: uproot file object
            The opened ROOT file.
    """

    def __init__(self, file_name : str, tree_name : str):
        ### Read file ###
        self.file_name = file_name
        self.tree_name = tree_name
        self.data = {}
        self.file = None
        uproot.default_library = 'awk'

    def load_branches(self, branches : list[str]) -> None:
        """
        Load specific branches from the ROOT file into the data dictionary.

        Parameters
        ----------
        branches: list
            A list of branch names to load from the ROOT file.
        """
        file = uproot.open(self.file_name)
        self.file = file
        tree = file[self.tree_name]
        
        for branch in branches:
            if branch not in self.data:
                self.data[branch] = tree[branch].array()

    def cuts_calculator(self, data : np.ndarray, cut_min : float = None, cut_max : float = None) -> tuple:
        """
        Apply cuts to the data based on the specified minimum and maximum values.

        Parameters
        ----------
        data: array-like
            The data to apply cuts to.
        cut_min: float, optional
            The minimum cut value.
        cut_max: float, optional
            The maximum cut value.

        Returns
        -------
        data_cut: array-like
            The data after applying cuts.
        cut: array-like
            A boolean array indicating which elements were kept.
        """
        ### Defines cuts ###
        if cut_min is None or cut_max is None:
            print("Cuts not defined!")

        cut = (cut_min < data) & (data < cut_max)
        data_cut = data[cut]

        return data_cut, cut

    def anti_cuts_calculator(self, data : np.ndarray, cut_min : float = None, cut_max : float = None) -> tuple:
        """
        Apply anti-cuts to the data based on the specified minimum and maximum values.

        Parameters
        ----------
        data: array-like
            The data to apply anti-cuts to.
        cut_min: float, optional
            The minimum cut value.
        cut_max: float, optional
            The maximum cut value.

        Returns
        -------
        data_cut: array-like
            The data after applying anti-cuts.
        cut: array-like
            A boolean array indicating which elements were kept.
        """
        ### Defines cuts ###
        if cut_min is None or cut_max is None:
            print("Cuts not defined!")

        cut = (cut_min > data) | (data > cut_max)
        data_cut = data[cut]

        return data_cut, cut

    def FT_Energy_corr(self, x : vec.array) -> vec.array:
        """
        CLAS12 Forward Tagger Energy correction

        Only works for Spring2019 RGA dataset!!!
        """

        E_cor = x.E + 0.085643 - 0.0288063*x.E + 0.00894691*pow(x.E, 2) - 0.000725449*pow(x.E, 3)

        Px_el = E_cor * (x.x / x.mag)
        Py_el = E_cor * (x.y / x.mag)
        Pz_el = E_cor * (x.z / x.mag)

        p_e_cor = vec.array({'x': Px_el, 'y': Py_el, 'z': Pz_el, 'E': E_cor})

        return p_e_cor
                
    def save_cuts(self, file_name: str, cuts) -> None:
        """
        Saves the applied cuts to a new ROOT file.

        Parameters
        ----------
        file_name: str
            The name of the output ROOT file.
        cuts: array-like
            The boolean array indicating which elements were kept.
        """

        if self.file is None:
            print("No file loaded!")
            return
        
        if self.file == file_name:
            print('New file cannot have same name as original!')
            return
        
        if cuts is None:
            print("No cuts applied! Aborting saving!")
            return
    
        # Open original tree
        tree = self.file[self.tree_name]
    
        # Save everything back into a new ROOT file
        with uproot.recreate(file_name) as save_file:
            save_file[self.tree_name] = {branch: tree[branch].array()[cuts] for branch in tree.keys()}



    def save_data(self, data: dict = None, cuts=None, file_name: str = None) -> None:
        """
        Save processed data to a ROOT file.

        Parameters
        ----------
        data: dict, optional
            A dictionary containing the data to save.
        cuts: array-like, optional
            A boolean array indicating which elements were kept.
        file_name: str, optional
            The name of the output ROOT file.
        """

        if self.file is None:
            print("No file loaded!")
            return

        if file_name is None:
            print("No output file name specified!")
            return

        if self.file == file_name:
            print("New file cannot have the same name as the original!")
            return

        # Open original tree
        tree = self.file[self.tree_name]

        with uproot.recreate(file_name) as save_file:
            # Save tree data (with cuts if given)
            if cuts is not None:
                save_file[self.tree_name] = {branch: tree[branch].array()[cuts] for branch in tree.keys()}
            else:
                save_file[self.tree_name] = {branch: tree[branch].array() for branch in tree.keys()}

            # Save additional data per Q2 bin
            if data is not None:
                for (q2min, q2max), dat in data.items():
                    q2_dict = {}
            
                    # Save each field under a directory named by Q2 bin
                    path = f'Q2_{q2min:.3f}_{q2max:.3f}'

                    if isinstance(dat, dict):
                        for key in dat.keys():
                            value = dat[key]
                            if key.startswith('p_'):
                                key_parts = key.split('_')
                                # Convert particle data to vector arrays if needed
                                q2_dict[f"px_{key_parts[1]}"] = value.px
                                q2_dict[f"py_{key_parts[1]}"] = value.py
                                q2_dict[f"pz_{key_parts[1]}"] = value.pz
                                q2_dict[f"M_{key_parts[1]}"] = value.M
                            else:
                                # For other data types, save directly
                                print(len(value), "value length for key:", key)
                                q2_dict[key] = ak.Array(value)
                    elif isinstance(dat, list):
                        # If data is a list, convert to awkward array
                        q2_dict = ak.Array(dat)
                    else:
                        print(f"Unsupported data type for Q2 bin [{q2min:.3f}, {q2max:.3f}]: {type(dat)}")
                        continue

                    save_file[path] = q2_dict

            else:
                print("No additional data to save!")

        print(f"Data saved to {file_name} successfully.")


# %%
