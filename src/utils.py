import os
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
from collections import OrderedDict
from typing import Tuple, List, Dict
from glob import glob
#################### torch_geometric module ####################

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge
from prettytable import PrettyTable

#################### torch_geometric module ####################


def cellArrayDivider(input_array):
    "Genearte faces for each cell in the input array."
    N = len(input_array)
    cursor = 0
    head_id = []
    segs = []

    while cursor < N:
        head_id.append(cursor)
        segs.append(input_array[cursor + 1 : cursor + input_array[cursor] + 1])
        cursor = cursor + input_array[cursor] + 1
    return segs

def vtp2Graph_poisson(input_file: str, data_label: List = ['source','temperature'])->Data:
    """
    The function reads .vtp file and converts it into a pyg mesh graph.
        
    Args:
        input_file (str): Path to vtp file.
        data_label (list of strings): By defalut ['source', 'temperature'],
    """
    mesh = pv.read(input_file)
    segs = cellArrayDivider(mesh.faces)  # face, triangle
    points = np.array(mesh.points, dtype=np.float32)  # pos
    transform = FaceToEdge(
        remove_faces=False
    )  ### undirected graph, meaning eij exists for every eji
    mesh_graph = Data(
        pos=torch.tensor(points),
        face=torch.tensor(np.array(segs).T),
    )

    for i in range(len(data_label)):
        temp_data = np.array(mesh.point_data[data_label[i]], dtype=np.float32)
        if len(temp_data.shape) == 1:
            temp_data = temp_data[..., None]
        mesh_graph[data_label[i]] = torch.tensor(temp_data)
    mesh_graph_transformed = transform(mesh_graph)
    return mesh_graph_transformed


def vtp2Graph_aorta(input_file: str) -> Tuple[Data, dict]:
    """
    The function reads .vtp file and converts it into a pyg mesh graph.
    Args:
    input_file (str): Path to the vtp file.

    Returns:
    tuple:
    - mesh_graph_transformed (Data): pyg Data object representing aorta.
    - readme (dict): information about the features and labels in the graph.
    """

    aorta_data = pv.read(input_file)

    segs = np.array(cellArrayDivider(aorta_data.faces)).T  # generate face
    points = np.array(aorta_data.points, dtype=np.float32)  # generate node pos

    nodal_normals = np.array(aorta_data.point_normals, dtype=np.float32)  # normalVector
    nodal_pressure = aorta_data.point_data["pressure"]  # pressure
    nodal_wss = aorta_data.point_data["wallShearStress"]  # wss

    nodal_stmdist = aorta_data.point_data["Abscissas"]  # dist from one point to inlet

    # nodal_ParallelTransport = aorta_data.point_data[
    #     "ParallelTransportNormals"
    # ]  # transport normals along a vessel centerline
    # nodal_SphereRadius = aorta_data.point_data["MaximumInscribedSphereRadius"]
    # nodal_tangent = aorta_data.point_data["FrenetTangent"]
    # nodal_curvature = aorta_data.point_data["Curvature"]
    # nodal_angle = aorta_data.point_data["AngularMetric"]
    nodal_labels = np.hstack(
        (
            np.array(nodal_pressure, dtype=np.float32)[:, np.newaxis],
            np.array(nodal_wss, dtype=np.float32),
        )
    )  # four dimensions

    transform = FaceToEdge(remove_faces=False)  # convert face to edge_index
    readme = {
        "pos": "corrdinatex, corrdinatey, corrdinatez",
        "y": "pressure, wssx, wssy, wssz",
        "norm": "normalx, normaly, normalz",
        "stmdist": "stmdist",
        "face": "triangles",
        "sphere_radius": "maximum inscribed sphere radius for each point along centerline",
        "edge_index": "graph connection",
        "paralleltransportNorm": "transport normals along a vessel centerline",
        "tanget": "FrenetTangent",
        "curvature": "curvature alone center line",
        "angle": "angle around centerline",
    }

    mesh_graph = Data(
        y=torch.tensor(nodal_labels, dtype=torch.float32),
        pos=torch.tensor(points, dtype=torch.float32),
        norm=torch.tensor(nodal_normals, dtype=torch.float32),
        stmdist=torch.tensor(
            np.array(nodal_stmdist, dtype=np.float32)[:, np.newaxis],
            dtype=torch.float32,
        ),
        face=torch.tensor(segs),
        # paralleltransportNorm=torch.tensor(
        #     nodal_ParallelTransport, dtype=torch.float32
        # ),
        # sphere_radius=torch.tensor(nodal_SphereRadius, dtype=torch.float32).unsqueeze(
        #     1
        # ),
        # tanget=torch.tensor(nodal_tangent, dtype=torch.float32),
        # curvature=torch.tensor(nodal_curvature, dtype=torch.float32).unsqueeze(1),
        # angle=torch.tensor(nodal_angle, dtype=torch.float32).unsqueeze(1),
    )

    mesh_graph_transformed = transform(mesh_graph)
    return mesh_graph_transformed, readme

class data_info(object):
    def __init__(
        self, path: str, output_path: str, labels=List
    ):

        self.path = path
        self.labels = labels
        self.output_path = output_path

        dataframe = self.graph2pandas()
        self.info = self.Get_data_info(dataframe)
        self.info.set_index('label', inplace=True) 

    def graph2pandas(self):
        r"convert pyg graph to pandas dataFrame"

        return NotImplementedError("To be implemented in subclass")

    def Get_data_info(self, dataframe=pd.DataFrame):
        """
        Function to read 3d centerline info
        """
        if os.path.exists(self.output_path + "/data_range_summary.csv"):
            print("csv already exists, loading data...")
            dfs = pd.read_csv(self.output_path + "/data_range_summary.csv")
            print("data load done.")
            return dfs
        else:
            print("Generating data info...")
            # Initialize a dictionary to hold the summary info for each label
            summary_data = {"label": [], "max": [], "min": [], "mean": [], "std": []}
            dfs = dataframe
            # Iterate over each label
            for label in tqdm(self.labels):
                # Load the data for the current label
                df = dfs[label]
                # Calculate max, min, mean, and std for each column
                for column in df.columns:
                    max_val = df[column].max()
                    min_val = df[column].min()
                    mean_val = df[column].mean()
                    std_val = df[column].std()

                    # Append the data info to the summary dictionary
                    summary_data["label"].append(column)
                    summary_data["max"].append(max_val)
                    summary_data["min"].append(min_val)
                    summary_data["mean"].append(mean_val)
                    summary_data["std"].append(std_val)

            # Create a summary DataFrame from the summary_data dictionary
            summary_df = pd.DataFrame(summary_data)
            # Save the summary DataFrame to a CSV file
            summary_df.to_csv(self.output_path + "/data_range_summary.csv", index=False)
            print(
                "data generation done, saved in directory:{}".format(self.output_path)
            )
            return summary_df

class poisson_info(data_info):
    def __init__(
        self,
        path: str,
        output_path: str,
        labels=List,
    ) -> None:
        super().__init__(path, output_path, labels)

    def graph2pandas(self):
        r"convert pyg graph to pandas dataFrame"
        categorized_data = {label: [] for label in self.labels}
        # Iterate over each .pt file in the folder
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            data = vtp2Graph_poisson(file_path)
            for label in self.labels:
                categorized_data[label].append(data[label])

        # Create dataframes according to the labels
        dfs = {}
        for label in self.labels:
            # Concatenate the list of tensors for the current label into a single tensor
            concatenated_tensor = torch.cat(categorized_data[label], dim=0)

            if label == "temperature":
                # Convert the tensor to a pandas DataFrame
                df = pd.DataFrame(
                    concatenated_tensor.numpy(),
                    columns=["temperature"],
                )
            elif label == "pos":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["x", "y", "z"])
            
            elif label == "source":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["source"])
            else:
                df = pd.DataFrame(concatenated_tensor.numpy())

            # Store the DataFrame in the dictionary
            dfs[label] = df
        return dfs

class aorta_3D_info(data_info):
    def __init__(
        self, path: str, output_path: str, labels=List
    ):
        super().__init__(path, output_path, labels)

    def graph2pandas(self) -> Dict:
        r"convert pyg data to pd dataFrame"

        categorized_data = {label: [] for label in self.labels}
        # Iterate over each .pt file in the folder
        for filename in os.listdir(self.path):
            file_path = os.path.join(self.path, filename)
            data, _ = vtp2Graph_aorta(file_path)
            for label in self.labels:
                categorized_data[label].append(data[label])

        # Create dataframes according to the labels
        dfs = {}
        for label in self.labels:
            # Concatenate the list of tensors for the current label into a single tensor
            concatenated_tensor = torch.cat(categorized_data[label], dim=0)

            if label == "y":
                # Convert the tensor to a pandas DataFrame
                df = pd.DataFrame(
                    concatenated_tensor.numpy(),
                    columns=["pressure", "wssx", "wssy", "wssz"],
                )
            elif label == "pos":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["x", "y", "z"])
            elif label == "stmdist":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["distance"])
            elif label == "norm":
                df = pd.DataFrame(
                    concatenated_tensor.numpy(), columns=["norm_x", "norm_y", "norm_z"]
                )
            elif label == "sphere_radius":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["sphere_radius"])
            elif label == "tanget":
                df = pd.DataFrame(
                    concatenated_tensor.numpy(), columns=["tan_x", "tan_y", "tan_z"]
                )
            elif label == "paralleltransportNorm":
                df = pd.DataFrame(
                    concatenated_tensor.numpy(),
                    columns=["trans_norm_x", "trans_norm_y", "trans_norm_z"],
                )
            elif label == "curvature":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["curvature"])
            elif label == "angle":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["angle"])
            else:
                df = pd.DataFrame(concatenated_tensor.numpy())

            # Store the DataFrame in the dictionary
            dfs[label] = df
        return dfs
    
## Especially for Suk's data, which contains multiple samples in one .pt file !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class suk_aorta_3D_info(data_info):
    def __init__(
        self, path: str, output_path: str, labels=List
    ):
        super().__init__(path, output_path, labels)

    def graph2pandas(self) -> Dict:
        r"convert pyg data to pd dataFrame"

        categorized_data = {label: [] for label in self.labels}
        # Iterate over each .pt file in the folder

        for fpath in sorted(glob(os.path.join(self.path, "data_*.pt"))):
            data: Data = torch.load(fpath, weights_only=False)

            for label in self.labels:
                categorized_data[label].append(data[label])

        # Create dataframes according to the labels
        dfs = {}
        for label in self.labels:

            # DEBUG: Print the label and the list of tensors
            print(f"Processing label: '{label}'")
            print(f"Number of tensors found: {len(categorized_data[label])}")
            
            # Concatenate the list of tensors for the current label into a single tensor
            concatenated_tensor = torch.cat(categorized_data[label], dim=0)

            if label == "y":
                # Convert the tensor to a pandas DataFrame
                df = pd.DataFrame(
                    concatenated_tensor.numpy(),
                    columns=["pressure", "wssx", "wssy", "wssz"],
                )
            elif label == "pos":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["x", "y", "z"])
            elif label == "stmdist":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["distance"])
            elif label == "norm":
                df = pd.DataFrame(
                    concatenated_tensor.numpy(), columns=["norm_x", "norm_y", "norm_z"]
                )
            elif label == "sphere_radius":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["sphere_radius"])
            elif label == "tanget":
                df = pd.DataFrame(
                    concatenated_tensor.numpy(), columns=["tan_x", "tan_y", "tan_z"]
                )
            elif label == "paralleltransportNorm":
                df = pd.DataFrame(
                    concatenated_tensor.numpy(),
                    columns=["trans_norm_x", "trans_norm_y", "trans_norm_z"],
                )
            elif label == "curvature":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["curvature"])
            elif label == "angle":
                df = pd.DataFrame(concatenated_tensor.numpy(), columns=["angle"])
            else:
                df = pd.DataFrame(concatenated_tensor.numpy())

            # Store the DataFrame in the dictionary
            dfs[label] = df
        return dfs

            
class Normalizer_ts:
    def __init__(self, params=[], method="-11", dim=None):
        self.params = params
        self.method = method
        self.dim = dim

    def fit_normalize(self, data):
        assert type(data) == torch.Tensor
        if len(self.params) == 0:
            if self.method == "-11" or self.method == "01":
                if self.dim == None:
                    self.params = (torch.max(data), torch.min(data))
                else:
                    self.params = (
                        torch.max(data, dim=self.dim)[0],
                        torch.min(data, dim=self.dim)[0],
                    )
            elif self.method == "ms":
                if self.dim == None:
                    self.params = (
                        torch.mean(data, dim=0),
                        torch.std(data, dim=self.dim),
                    )
                else:
                    self.params = (
                        torch.mean(data, dim=self.dim),
                        torch.std(data, dim=self.dim),
                    )

        return self.fnormalize(data, self.params, self.method)

    def normalize(self, new_data):
        return self.fnormalize(new_data, self.params, self.method)

    def denormalize(self, new_data_norm):
        return self.fdenormalize(new_data_norm, self.params, self.method)

    def get_params(self):
        if self.method == "ms":
            print("returning mean and std")
        if self.method == "-11" or self.method == "01":
            print("returning max and min")
        return self.params

    @staticmethod
    def fnormalize(data, params, method):
        if method == "-11":
            return (data - params[1]) / (params[0] - params[1]) * 2 - 1
        if method == "ms":
            return (data - params[0]) / params[1]
        if method == "01":
            return (data - params[1]) / (params[0] - params[1])

    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == "-11":
            return (data_norm + 1) * 2 * (params[0] - params[1]) + params[1]
        if method == "ms":
            return data_norm * params[1] + params[0]
        if method == "01":
            return (data_norm) * (params[0] - params[1]) + params[1]



# Get rid of the prefix "module." in the state dict
def parallel_to_serial(ordered_dict):
    return OrderedDict((key[7:], value) for key, value in ordered_dict.items())


# Return ordered state dictionary for serial data model
def load(path, map_location):
    ordered_dict = torch.load(path, map_location)
    if next(iter(ordered_dict)).startswith("module."):
        return parallel_to_serial(ordered_dict)
    else:
        return ordered_dict


def parameter_table(model):
    table = PrettyTable(["Modules", "Parameters"])
    total = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total += params
    table.add_row(["TOTAL", total])

    return table