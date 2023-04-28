#---------------------------Import Packages------------------------------------

import os
import numpy as np
import json

def query_kwargs(key, default, **kwargs):
   return kwargs[key] if key in kwargs.keys() else default

#---------------------------Librarian------------------------------------
"""
used for loading creature data from the class of step size comparison
"""
class Librarian():

    def __init__(self, **kwargs):

        self.path = os.path.join(*os.path.abspath(__file__).split("/")[:-2])
        self.zoo_path = os.path.join('/', self.path, 'zoo')

        self.zoo_directory = query_kwargs("directory", self.zoo_path, **kwargs) 

        self.verbose = query_kwargs("verbose", True, **kwargs)

        self.update_index()

    def update_index(self):
        """
        update the list of (string) names of patterns in the zoo directory
        """

        pattern_names = os.listdir(self.zoo_directory)

        remove_list = []
        # filtre unwanted file
        for elem in pattern_names:
            if ".py" in elem \
                    or ".md" in elem \
                    or ".ipynb" in elem \
                    or "csv" in elem \
                    or "__pycache__" in elem:
                remove_list.append(elem)
                
        for elem in remove_list:
            pattern_names.remove(elem)

        pattern_names = [os.path.splitext(elem)[0] for elem in pattern_names]

        pattern_names.sort()

        self.index = pattern_names

    # store the file of creature's parameters
    # def store(self, pattern: np.array, pattern_name: str = "my_pattern",\
    #         config_name: str = "unspecified", entry_point="not specified",\
    #         commit_hash="not_specified", notes=None):

    #     counter = 0
    #     file_path = os.path.join(self.directory, f"{pattern_name}{counter:03}.npy")

    #     while os.path.exists(file_path):
    #         counter += 1
    #         file_path = os.path.join(self.directory, f"{pattern_name}{counter:03}.npy")

    #         if counter >= 1000:
    #             # shouldn't be here, assuming less than 1000 patterns of same name
    #             print(f"more than {counter} variants of pattern {pattern_name}, "\
    #                     f"consider choosing a new name.")

    #     meta_path = os.path.join(self.directory, 
    #             f"{pattern_name}{counter:03}.csv")


    #     if config_name == "unspecified" and self.verbose:
    #         print(f"warning, no config supplied for {pattern_name}")

    #     with open(meta_path, "w") as f:
    #         f.write(f"ca_config,{config_name}")
    #         f.write(f"\ncommit_hash,{commit_hash}")
    #         f.write(f"notes, {notes}")
    #         f.write(f"\nentry_point,{entry_point}")

    #     np.save(file_path, pattern) 

    #     if self.verbose:
    #         print(f"pattern {pattern_name} saved to {file_path}")
    #         print(f"pattern {pattern_name} metadata saved to {meta_path}")

    #     self.index.append(f"{pattern_name}  {counter:03}")

    def load(self, pattern_name: str) -> tuple([np.array, str]):
        """
        load pattern from disk
        """

        file_path = os.path.join(self.zoo_directory, f"{pattern_name}.json")

        with open(file_path, "r") as read_file:

            pattern = json.load(read_file)

        if self.verbose:
            print(f"pattern {pattern_name} loaded from {file_path}")

        return pattern

if __name__ == "__main__":

    librarian = Librarian()