import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.data import Dataset
import pandas as pd
import deepchem as dc
from rdkit import Chem
from tqdm import tqdm
import os

class engine: # Controls the NN training and testing stages.
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs): # Loss function for back propagation
        return nn.MSELoss()(outputs, targets)

    def train(self, data_loader): # Training of model, weights optimized with backpropagation here
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader) # Train loss

    def validate(self, data_loader): # Validation of model, trained weights are frozen and not touched during this process
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
                final_loss += loss.item()
        return final_loss / len(data_loader) # Valid loss

    def test(self, data_loader): # Testing model to evaluate model's performance
        self.model.eval()
        rmse_total = 0
        mae_total = 0
        mse_total = 0
        r2_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
                loss = loss.item()
                rmse = np.sqrt(loss)
                
                rmse_total += rmse
                
                mae = mean_absolute_error(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy()
                )
                mae_total += mae

                mse = mean_squared_error(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy()
                )
                mse_total += mse

                r2 = r2_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy()
                )
                r2_total += r2

        return (
            rmse_total / len(data_loader),
            mae_total / len(data_loader),
            mse_total / len(data_loader),
            r2_total / len(data_loader),
        ) # Performance of model with "human" comprehensible metrics


class engine_no_edge:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.MSELoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
                final_loss += loss.item()
        return final_loss / len(data_loader)

    def test(self, data_loader):
        self.model.eval()
        rmse_total = 0
        mae_total = 0
        mse_total = 0
        r2_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
                loss = loss.item()
                rmse = np.sqrt(loss)
                
                rmse_total += rmse

                mae = mean_absolute_error(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy()
                )
                mae_total += mae

                mse = mean_squared_error(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy()
                )
                mse_total += mse

                r2 = r2_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy()
                )
                r2_total += r2

        return (
            rmse_total / len(data_loader),
            mae_total / len(data_loader),
            mse_total / len(data_loader),
            r2_total / len(data_loader),
        )


class LipoFeatures(Dataset): # Dataset class is a general class that contains a few functions to help convert mols to features.
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(LipoFeatures, self).__init__(root, transform, pre_transform)
        # calls the parent class (Dataset) allocate variables root, transform, and pre-transform.
        # parent class calls some of the functions listed below from its __init__ function.
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        # what this means is that if we do not have the raw dataset available, we can call the download function to download the appropriate dataset.
        # however, since we have the data in the raw folder, we will not use this function.
        # this part is not coded as a result.
        return self.filename
    
    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        # skips running the process function if the processed directory has data inside.
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
 
    # key function here to process the data into features, called by parent class.
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        # this variable is used to convert molecules to its respective features. Can hover mouse over and read description.
        
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"]) #Change the col name repsectively
            #convert smiles to mol
            
            f = featurizer._featurize(mol)
            #convert mol to features
            
            data = f.to_pyg_graph()
            #python object modeling a single graph with various attributes (features)
            #gives a graph representation for the molecule and includes their respective features (edge and node) on the graph.
            
            data.y = self._get_label(row["LogD"]) #Change the col name repsectively
            #adds a label to the graphical representation, telling us the expected value of the graph.
            
            data.smiles = row["smiles"] #Change the col name repsectively
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data


class LogPFeatures(Dataset): #Dataset class is a general class that contains a few functions to help convert mols to features.
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(LogPFeatures, self).__init__(root, transform, pre_transform)
        #calls the parent class (Dataset) allocate variables root, transform, and pre-transform.
        #parent class calls some of the functions listed below from its __init__ function.
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        #what this means is that if we do not have the raw dataset available, we can call the download function to download the appropriate dataset.
        #however, since we have the data in the raw folder, we will not use this function.
        #this part is not coded as a result.
        return self.filename
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        return [f"data_{i}.pt" for i in list(self.data.index)]
        
    def download(self):
        pass
    #pass as download of raw data is not required since we have it. Can code it if we want to download file from somewhere.
    
    #key function here to process the data into features, called by parent class.
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        #this variable is used to convert molecules to its respective features. Can hover mouse over and read description.
        
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"]) #Change the col name repsectively
            #convert smiles to mol
            
            f = featurizer._featurize(mol)
            #convert mol to features
            
            data = f.to_pyg_graph()
            #python object modeling a single graph with various attributes (features)
            #gives a graph representation for the molecule and includes their respective features (edge and node) on the graph.
            
            data.y = self._get_label(row["LogP"]) #Change the col name repsectively
            #adds a label to the graphical representation, telling us the expected value of the graph.
            
            data.smiles = row["smiles"] #Change the col name repsectively
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
                
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                  f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                  f'data_{idx}.pt'))        
        return data
