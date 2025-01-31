from GNN_utils import engine_no_edge, LipoFeatures
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
import optuna
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import numpy as np

'''
Function
'''
class GIN(torch.nn.Module):

    def __init__(self, num_features, num_targets, num_layers, hidden_size):
        super(GIN, self).__init__()
        layers = []
        for _ in range(num_layers):
            if len(layers) == 0:
                layers.append(
                    GINConv(
                        Sequential(
                            Linear(num_features, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )
            else:
                layers.append(
                    GINConv(
                        Sequential(
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                            Linear(hidden_size, hidden_size),
                            ReLU(),
                        )
                    )
                )
        self.model = nn.Sequential(*layers)
        # Readout MLP

        self.ro = Sequential(
            Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, num_targets)
        )

    def forward(self, x, edge_index, batch_index):
        
        for layer in self.model:
            x = layer(x, edge_index)
        x = global_add_pool(x, batch_index)
        
        return self.ro(x)


def run_tuning(train_loader, valid_loader, params):

    model = GIN(num_features=NUM_FEATURES, num_targets=NUM_TARGET, num_layers=params['num_layers'], hidden_size=params['hidden_size'])
    model.to(DEVICE) 
    optimizer=torch.optim.Adam(model.parameters(),lr = params['learning_rate']) 
    eng = engine_no_edge(model, optimizer, device=DEVICE) 

    best_loss = np.inf 
    early_stopping_iter = PATIENCE 
    early_stopping_counter = 0  

    for epoch in range(EPOCHS): 
        train_loss = eng.train(train_loader) 
        valid_loss = eng.validate(valid_loader) 
        print(f'Epoch: {epoch+1}/{EPOCHS}, train loss : {train_loss}, validation loss : {valid_loss}') 
        if valid_loss < best_loss:
            best_loss = valid_loss  
            early_stopping_counter=0 #reset counter
        else:
           early_stopping_counter +=1

        if early_stopping_counter > early_stopping_iter:
            print('Early stopping...')
            break 
        print(f'Early stop counter: {early_stopping_counter}')
    
    return best_loss 


def run_training(train_loader, valid_loader, params, trained_model_path):
    model = GIN(num_features=NUM_FEATURES, num_targets=NUM_TARGET, num_layers=params['num_layers'], hidden_size=params['hidden_size'])
    model.to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine_no_edge(model, optimizer, device=DEVICE)

    best_loss = np.inf
    early_stopping_iter = PATIENCE
    early_stopping_counter = 0 

    for epoch in range(EPOCHS):
        train_loss= eng.train(train_loader)
        valid_loss= eng.validate(valid_loader)
        print(f'Epoch: {epoch+1}/{EPOCHS}, train loss : {train_loss}, validation loss : {valid_loss}')
        if valid_loss < best_loss:
            best_loss = valid_loss 
            early_stopping_counter=0 #reset counter
            print('Saving model...')
            torch.save(model.state_dict(), trained_model_path)
        else:
            early_stopping_counter +=1

        if early_stopping_counter > early_stopping_iter:
            print('Early stopping...')
            break
        print(f'Early stop counter: {early_stopping_counter}')
    
    return best_loss


def run_testing(test_loader, params, trained_model_path):
    model = GIN(num_features=NUM_FEATURES, num_targets=NUM_TARGET, num_layers=params['num_layers'], hidden_size=params['hidden_size'])
    model.load_state_dict(torch.load(trained_model_path))
    model.to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine_no_edge(model, optimizer, device=DEVICE)

    print('Begin testing...')
    rmse, mae, mse, r2 = eng.test(test_loader)
    print('Test completed!')
    print(f'rmse :{rmse}, mae:{mae}, mse :{mse}, r2: {r2}')
    return rmse, mae, mse, r2


'''
Model Tuning and Testing
'''
NUM_FEATURES=30
NUM_TARGET = 1
DEVICE = 'cuda'
PATIENCE = 10
EPOCHS = 300

def objective(trial):
    
    params = {
        'num_layers' : trial.suggest_int('num_layers', 1,3),
        'hidden_size' : trial.suggest_int('hidden_size', 64, 512),
        'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 9e-3, log=True)
    }

    dataset_for_cv = LipoFeatures(root='./Processed data/graph data/lipo_train', 
                                  filename='raw_data_train.csv')

    kf = KFold(5)
    fold_rmse = 0

    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(dataset_for_cv)):
        print(f'Fold {fold_no}')
        train_dataset= []
        valid_dataset = []
        for t_idx in train_idx:
            train_dataset.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{t_idx}.pt'))

        for v_idx in valid_idx:
            valid_dataset.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{v_idx}.pt'))
            
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

        loss = run_tuning(train_loader, valid_loader, params)
        fold_rmse += loss
        
    return fold_rmse/5

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=30)
trial_ = study.best_trial
best_params = trial_.params
print(f'best trial:{trial_.values}')
print(f'Best parameters: {trial_.params}')

# best trial:[0.6463802520717893]
# Best parameters: {'num_layers': 2, 'hidden_size': 400, 'learning_rate': 0.001546086980582617}

n_repetitions = 5
path_to_save_trained_model = './Models/gin_models'
rmse_list = []
mae_list = []
mse_list = []
r2_list = []

#load dataset 
dataset_for_cv = LipoFeatures(root='./Processed data/graph data/lipo_train', filename='raw_data_train.csv')
test_dataset = LipoFeatures(root='./Processed data/graph data/lipo_test', filename='raw_data_test.csv')
kf = KFold(n_splits= 5)

for repeat in range(n_repetitions):
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(dataset_for_cv)):
        print(f'For rep: {repeat}, fold: {fold_no}')
        train_dataset= []
        valid_dataset = []
        for t_idx in train_idx:
            train_dataset.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{t_idx}.pt'))
        for v_idx in valid_idx:
            valid_dataset.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{v_idx}.pt'))

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        run_training(train_loader, valid_loader, params, os.path.join(path_to_save_trained_model, f'gin_repeat_{repeat}_fold_{fold_no}.pt'))
        rmse, mae, mse, r2 = run_testing(test_loader, params, os.path.join(path_to_save_trained_model, f'gin_repeat_{repeat}_fold_{fold_no}.pt'))
        rmse_list.append(rmse)
        mae_list.append(mae)
        mse_list.append(mse)
        r2_list.append(r2)

rmse_arr = np.array(rmse_list)
mean_rmse = np.mean(rmse_arr)
sd_rmse = np.std(rmse_arr)
print(f'mae:{mean_rmse:.3f}±{sd_rmse:.3f}')

mae_arr = np.array(mae_list)
mean_mae = np.mean(mae_arr)
sd_mae = np.std(mae_arr)
print(f'mae:{mean_mae:.3f}±{sd_mae:.3f}')

mse_arr = np.array(mse_list)
mse_mean= np.mean(mse_arr)
mse_sd = np.std(mse_arr)
print(f'mse:{mse_mean:.3f}±{mse_sd:.3f}')

r2_arr = np.array(r2_list)
r2_mean= np.mean(r2_arr)
r2_sd = np.std(r2_arr)
print(f'r2: {r2_mean:.3f}±{r2_sd:.3f}')

# rmse:0.920±0.379
# mae:0.716±0.306
# mse:0.991±0.955
# r2: 0.690±0.298
