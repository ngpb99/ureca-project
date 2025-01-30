import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import optuna
import os
from GNN_utils import engine, LipoFeatures

''' 
Functions
'''
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
    model = GraphTrans(num_features=30, num_targets=1, 
                       num_layers=params['num_layers'], hidden_size=params['hidden_size'], 
                       n_heads=params['n_heads'], dropout=params['dropout'], edge_dim=11)
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

def model_training(params, train_loader, valid_loader, trained_model_path):
    model = GraphTrans(num_features = 30, num_targets = 1, 
                       num_layers = params['num_layers'], hidden_size = params['hidden_size'], 
                       n_heads = params['n_heads'], dropout = params['dropout'], edge_dim = 11)
    
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = 'cuda')
    
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    
    for epoch in range(300):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        print(f'For epoch {epoch+1}, Train loss is {train_loss}, Valid loss is {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), trained_model_path)
        
        else:
            early_stopping_counter += 1
            print(f'Early stopping counter: {early_stopping_counter}')
        
        if early_stopping_counter > early_stopping_iter:
            print('Commencing early stopping')
            print('Saving final model')
            break

def model_testing(params, test_loader, trained_model_path):
    model = GraphTrans(num_features = 30, num_targets = 1, 
                       num_layers = params['num_layers'], hidden_size = params['hidden_size'], 
                       n_heads = params['n_heads'], dropout = params['dropout'], edge_dim = 11)
    
    model.load_state_dict(torch.load(trained_model_path))
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = 'cuda')
    rmse, mae, mse, r2 = eng.test(test_loader)
    
    return rmse, mae, mse, r2

'''
Model Tuning
'''
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

'''
Model Training and Testing
'''
path_to_save_trained_model = './Models/gt_models'
best_params = {'num_layers': 2, 'hidden_size': 111, 'n_heads': 7, 
               'dropout': 0.31119124560085437, 'learning_rate': 0.0011599554736897006}

idx_for_cv = LipoFeatures(root = './Processed data/graph data/lipo_train',
                          filename = 'raw_data_train.csv')
test_idx = LipoFeatures(root = './Processed data/graph data/lipo_test', 
                        filename = 'raw_data_test.csv')
test_list = []
for i in range(len(test_idx)):
    test_list.append(torch.load(f'./Processed data/graph data/lipo_test/processed/data_{i}.pt'))
test_loader = DataLoader(test_list, batch_size=256, shuffle=False)
kf = KFold(n_splits = 5)

rmse_list = []
mae_list = []
mse_list = []
r2_list = []

for repeats in range(5):
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(idx_for_cv)):
        print(f'Commencing repetition {repeats+1}, Fold no. {fold_no + 1}')
        train_list = []
        valid_list = []
        
        for t in train_idx:
            train_list.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{t}.pt'))
        for v in valid_idx:
            valid_list.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{v}.pt'))
        
        
        train_loader = DataLoader(train_list, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_list, batch_size=256, shuffle=False)
        
        model_training(params = best_params, train_loader = train_loader, 
                   valid_loader = valid_loader, trained_model_path = os.path.join(path_to_save_trained_model, f'GT_repeat_{repeats}_fold_{fold_no}.pt'))
        rmse, mae, mse, r2 = model_testing(params = best_params, test_loader = test_loader,
                                         trained_model_path = os.path.join(path_to_save_trained_model, f'GT_repeat_{repeats}_fold_{fold_no}.pt'))
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        mse_list.append(mse)
        r2_list.append(r2)
        print(f'Repetition no. {repeats+1}, Fold no. {fold_no + 1} has been completed! \n')

rmse_array = np.array(rmse_list)
mae_array = np.array(mae_list)
mse_array = np.array(mse_list)
r2_array = np.array(r2_list)

rmse_mean = np.mean(rmse_array)
mae_mean = np.mean(mae_array)
mse_mean = np.mean(mse_array)
r2_mean = np.mean(r2_array)

rmse_sd = np.std(rmse_array)
mae_sd = np.std(mae_array)
mse_sd = np.std(mse_array)
r2_sd = np.std(r2_array)

print(f'RMSE:{rmse_mean:.3f}±{rmse_sd:.3f}')
print(f'MAE:{mae_mean:.3f}±{mae_sd:.3f}')
print(f'MSE:{mse_mean:.3f}±{mse_sd:.3f}')
print(f'r2:{r2_mean:.3f}±{r2_sd:.3f}')

# RMSE:0.874 ± 0.073
# MAE:0.681 ± 0.062
# MSE:0.771 ± 0.132
# r2:0.759 ± 0.041
