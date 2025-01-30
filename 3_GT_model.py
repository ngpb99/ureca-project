import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import optuna
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem

''' 
Functions
'''
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
        return torch.tensor(label, dtype=torch.int64)

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

class GraphTrans(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers,
        hidden_size,
        n_heads,
        dropout,
        edge_dim,
    ):
        super(GraphTrans, self).__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(
                    TransformerConv(
                        num_features,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )
            else:
                layers.append(
                    TransformerConv(
                        hidden_size * n_heads,
                        hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        beta=True,
                    )
                )

        self.model = nn.Sequential(*layers)
        # Readout MLP
        self.ro = Sequential(
            Linear(hidden_size * n_heads, hidden_size),
            ReLU(),
            Linear(hidden_size, num_targets),
        )

    def forward(self, x, edge_attr, edge_index, batch_index):
        for layer in self.model:
            x = layer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch_index)
        return self.ro(x)

def run_tuning(train_loader, valid_loader, params):
    model = GraphTrans(num_features=NUM_FEATURES, num_targets=NUM_TARGET, 
                       num_layers=params['num_layers'], hidden_size=params['hidden_size'], 
                       n_heads=params['n_heads'], dropout=params['dropout'], edge_dim=EDGE_DIM)
    model.to('cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine(model, optimizer, device='cuda')

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0 

    for epoch in range(300):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        print(f'Epoch: {epoch+1}/300, train loss : {train_loss}, validation loss : {valid_loss}')
        if valid_loss < best_loss:
            best_loss = valid_loss 
            early_stopping_counter=0
        else:
            early_stopping_counter +=1

        if early_stopping_counter > early_stopping_iter:
            print('Early stopping...')
            break
        print(f'Early stop counter: {early_stopping_counter}')
    
    return best_loss

'''
Model Tuning
'''
NUM_FEATURES=30
NUM_TARGET = 1
EDGE_DIM = 11
DEVICE = 'cuda'
SEED_NO = 40

def objective(trial):
    params = {
        'num_layers' : trial.suggest_int('num_layers', 1,3),
        'hidden_size' : trial.suggest_int('hidden_size', 64, 512),
        'n_heads' : trial.suggest_int('n_heads', 1, 5),
        'dropout': trial.suggest_float('dropout', 0.1,0.4),
        'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 9e-3, log=True)
    }
    
    #load dataset 
    dataset_for_cv = LipoFeatures(root='./Processed data/graph data/lipo_train', 
                        filename='raw_data_train.csv')
    
    kf = KFold(n_splits=5)
    fold_loss = 0

    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(dataset_for_cv)):
        print(f'Fold {fold_no}')
        train_dataset= []
        valid_dataset = []
        for t_idx in train_idx:
            train_dataset.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{t_idx}.pt'))
        for v_idx in valid_idx:
            valid_dataset.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{v_idx}.pt'))
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

        loss = run_tuning(train_loader, valid_loader, params)
        fold_loss += loss

    return fold_loss/5

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=30)
trial_ = study.best_trial
print(f'best trial:{trial_.values}')
print(f'Best parameters: {trial_.params}')
  
