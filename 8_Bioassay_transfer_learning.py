import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from GNN_utils import VerticalModel_GT_GIN_CombinedEmbeds, Bioassay, engine, engine_classsification, LipoFeatures
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from timeit import default_timer as timer
import torch
import numpy as np
import optuna
import os

'''
Function
'''
class VerticalModel_GT_GIN_CombinedEmbeds(torch.nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, heads, 
                 dropout, edge_dim, num_layers_GIN, num_layers_GT):
        
        super(VerticalModel_GT_GIN_CombinedEmbeds, self).__init__()
        
        gt_layers = []
        for layers in range(num_layers_GT):
            if layers == 0:
                gt_layers.append(TransformerConv(in_channels = num_features, 
                                                 out_channels = hidden_size,
                                                 heads = heads,
                                                 dropout = dropout,
                                                 edge_dim = edge_dim,
                                                 beta = True))
            else:
                gt_layers.append(TransformerConv(in_channels = hidden_size * heads, 
                                                 out_channels = hidden_size,
                                                 heads = heads,
                                                 dropout = dropout,
                                                 edge_dim = edge_dim,
                                                 beta = True))
        
        gin_layers = []
        for layers in range(num_layers_GIN):
            if layers == 0:
                gin_layers.append(GINConv(Sequential(
                    Linear(hidden_size * heads, hidden_size),
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
            
            else:
                gin_layers.append(GINConv(Sequential(
                    Linear(hidden_size, hidden_size),
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
        
        self.gt_model = nn.Sequential(*gt_layers)
        self.gin_model = nn.Sequential(*gin_layers)
        
        self.readout = Sequential(Linear(hidden_size + hidden_size * heads, hidden_size),
                                  ReLU(),
                                  Linear(hidden_size, num_targets))
        
    def forward(self, x, edge_attr, edge_index, batch):
        
        all_embeds = []
        for layer_no, layers in enumerate(self.gt_model):
            if layer_no == 0:
                gt_embed = layers(x, edge_index, edge_attr)
            else:
                gt_embed = layers(gt_embed, edge_index, edge_attr)
        
        gt_final = global_mean_pool(gt_embed, batch)
        all_embeds.append(gt_final)
        
        for layer_no, layers in enumerate(self.gin_model):
            if layer_no == 0:
                gin_embed = layers(gt_embed, edge_index)
            else:
                gin_embed = layers(gin_embed, edge_index)
                
        gin_final = global_add_pool(gin_embed, batch)
        all_embeds.append(gin_final)
        
        concat_embeds = torch.concat(all_embeds, dim = 1)
        output = self.readout(concat_embeds)
        
        return output


def run_tuning(params, train_loader, valid_loader):
    model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    eng = engine(model, optimizer, DEVICE)
    best_loss = np.inf
    early_stopping_counter = 0
    
    for epoch in range(EPOCH):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        print(f'Epoch: {epoch+1}/{EPOCH}, train_loss: {train_loss}, valid_loss: {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
        
        else:
            early_stopping_counter += 1
            print(f'Early stop counter: {early_stopping_counter}')
        
        if early_stopping_counter > PATIENCE:
            print('commencing early stopping...')
            break
        
    return best_loss


def run_training(params, train_loader, valid_loader, trained_model_path, epoch):
    model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.3, patience = 20)
    eng = engine(model, optimizer, DEVICE)
    
    for epoch in range(EPOCH):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        scheduler.step(valid_loss)
        print(f'For {epoch+1}/{EPOCH}, train loss is {train_loss}, valid loss is {valid_loss}')
        end_timer = timer()
        
        if epoch + 1 == EPOCH:
            print('1000 epochs have been completed, saving model...')
            torch.save(model.state_dict(), trained_model_path)
        
        if (end_timer - start_timer) > 129600: #36 hours
            print('commence early stopping as time limit has reached, saving model...')
            torch.save(model.state_dict(), trained_model_path)
            break
            
    return valid_loss


def model_finetuning(params, train_loader, valid_loader, pretrained_model_path, trained_model_path):
    model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(DEVICE)
    
    # fine tuning at 0.1 x lr
    new_parameters = [{'params': model.gin_model.parameters(), 'lr': params['learning_rate']/10}, 
                      {'params': model.gt_model.parameters(), 'lr': params['learning_rate']/10},
                      {'params': model.readout.parameters(), 'lr': params['learning_rate']}]
        
    optimizer = torch.optim.Adam(new_parameters)
    eng = engine(model = model, optimizer = optimizer, device = DEVICE)
    best_loss = np.inf
    early_stopping_counter = 0
    
    for epoch in range(EPOCH):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        
        print(f'For epoch {epoch+1}/{EPOCH}, train loss: {train_loss}, valid loss: {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), trained_model_path)
            print('saving model...')
        
        else:
            early_stopping_counter += 1
            print(f'early stopping counter: {early_stopping_counter}')
        
        if early_stopping_counter > PATIENCE:
            print('commencing early stopping...')
            break
    
    return best_loss


def model_testing(params, test_loader, trained_model_path):
    model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    
    model.load_state_dict(torch.load(trained_model_path))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = DEVICE)
    
    rmse, mae, mse, r2 = eng.test(test_loader)
    
    return rmse, mae, mse, r2

'''
Model Tuning with Bioassay
'''
NUM_TARGETS = 128 # Dataset is multiclass
NUM_FEATURES = 30
EDGE_DIM = 11
DEVICE = 'cuda'
EPOCH = 300
PATIENCE = 10
N_SPLITS = 5

def objective(trial):
    params = {'hidden_size': trial.suggest_int('hidden_size', 64, 512),
              'num_layers_GIN': trial.suggest_int('num_layers_GIN', 1, 3),
              'num_layers_GT': trial.suggest_int('num_layers_GT', 1, 3),
              'dropout': trial.suggest_float('dropout', 0.1, 0.4),
              'heads': trial.suggest_int('heads', 5, 10),
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 9e-3, log=True)}
    
    PCBA_data = Bioassay(root = './Processed data/graph data/PCBA', 
                             filename = 'PCBA_finalized_data.csv')
    
    kf = KFold(n_splits=N_SPLITS)
    total_loss = 0
    
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(PCBA_data)):
        print(f'Fold No.:{fold_no}')
        train_list = []
        valid_list = []
        for t in train_idx:
            train_list.append(torch.load(f'./Processed data/graph data/PCBA/processed/data_{t}.pt'))
        for v in valid_idx:
            valid_list.append(torch.load(f'./Processed data/graph data/PCBA/processed/data_{v}.pt'))
        
        train_loader = DataLoader(train_list, batch_size = 512, shuffle = True)
        valid_loader = DataLoader(valid_list, batch_size = 512, shuffle = False)
        
        run_loss = run_tuning(params, train_loader, valid_loader)
        total_loss += run_loss
    
    return (total_loss/N_SPLITS)

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 20)
trial_ = study.best_trial
print(f'best trial:{trial_.values}')
print(f'Best parameters: {trial_.params}')

'''
Pretraining Model
'''
DEVICE = 'cuda'
NUM_FEATURES = 30
NUM_TARGETS = 128
EDGE_DIM = 11
model_directory = './Models/bioassay_pretrained_models'
best_params = {'hidden_size': 125, 'num_layers_GIN': 3, 'num_layers_GT': 1, 'dropout': 0.2062818343772681, 'heads': 6, 'learning_rate': 0.0012646699060966755}
EPOCH = 1000

end_idx_train = 393913 # 0.1 split fraction (0.9 for train, 0.1 for valid), manual split.. could have used sklearn's train_test_split
start_idx_test = end_idx_train
end_idx_test = 437682 # length of PCBA

train_list = []
valid_list = []
for t in range(end_idx_train):
    train_list.append(torch.load(f'./Processed data/graph data/PCBA/processed/data_{t}.pt'))
for v in range(start_idx_test, end_idx_test):
    valid_list.append(torch.load(f'./Processed data/graph data/PCBA/processed/data_{v}.pt'))
    
train_loader = DataLoader(train_list, batch_size = 512, shuffle = True)
valid_loader = DataLoader(valid_list, batch_size = 512, shuffle = False)

# K-Fold cross validation is removed at pretraining stage as it tells us the average performance of model with different train/test datapoints.
# Computationally redundant as I do not need to know how well the model performs at predicting Biosassay.
# Follow paper to pretrain model for 1000 epochs instead of early stopping.

start_timer = timer()
run_loss = run_training(params = best_params, train_loader = train_loader, valid_loader = valid_loader, 
                              trained_model_path = os.path.join(model_directory, 'pretrained_pcba_1000_epochs_with_scheduler.pt'), 
                              epoch = EPOCH)
print(f'Run loss (MSE) for pretrained model is {run_loss}')

'''
Changing readout layer from 128 (multiclass) to 1 (regression)
'''
best_params = {'hidden_size': 125, 'num_layers_GIN': 3, 'num_layers_GT': 1, 'dropout': 0.2062818343772681, 'heads': 6, 'learning_rate': 0.0012646699060966755}
model = VerticalModel_GT_GIN_CombinedEmbeds(num_features = 30, num_targets = 128, 
                                            hidden_size = best_params['hidden_size'], heads = best_params['heads'], 
                                            dropout = best_params['dropout'], edge_dim = 11, 
                                            num_layers_GIN = best_params['num_layers_GIN'], num_layers_GT = best_params['num_layers_GT'])

model.load_state_dict(torch.load('./Models/bioassay_pretrained_models/pretrained_pcba_1000_epochs_with_scheduler.pt'))
new_num_target = 1
new_readout = Sequential(Linear(best_params['hidden_size'] + best_params['hidden_size'] * best_params['heads'], best_params['hidden_size']),
                         ReLU(),
                         Linear(best_params['hidden_size'], new_num_target)) #Changing readout from 128 dimensions to 1 dimension

model.readout = new_readout
trained_model_path = './Models/bioassay_pretrained_models/pretrained_pcba_readout_layer_changed.pt'
torch.save(model.state_dict(), trained_model_path)

'''
Finetuning Model
'''
NUM_FEATURES = 30
NUM_TARGETS = 1
EDGE_DIM = 11
DEVICE = 'cuda'
EPOCH = 300
PATIENCE = 10
N_SPLITS = 5
N_REPETITION = 5
best_params = {'hidden_size': 125, 'num_layers_GIN': 3, 'num_layers_GT': 1, 'dropout': 0.2062818343772681, 'heads': 6, 'learning_rate': 0.0012646699060966755}

model_name = 'pretrained_pcba_readout_layer_changed.pt'
pretrained_model_directory = './Models/bioassay_pretrained_models'
model_directory = './Models/finetuned_bioassay_pretrained_models'

LogD_train_data = LipoFeatures(root = './Processed data/graph data/lipo_train', 
                         filename = 'raw_data_train.csv')
LogD_test_data = LipoFeatures(root = './Processed data/graph data/lipo_test',
                              filename = 'raw_data_test.csv')
test_loader = DataLoader(LogD_test_data, batch_size = 256, shuffle = False)

kf = KFold(n_splits = N_SPLITS)
rmse_list = []
mae_list = []
mse_list = []
r2_list = []

for repeat in range(N_REPETITION):
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(LogD_train_data)):
        print(f'Commencing repetition {repeat+1}, Fold no. {fold_no + 1}')
        train_list = []
        valid_list = []
        
        for t in train_idx:
            train_list.append(torch.load(f'./graph data/lipo_train/processed/data_{t}.pt'))
        for v in valid_idx:
            valid_list.append(torch.load(f'./graph data/lipo_train/processed/data_{v}.pt'))
            
        train_loader = DataLoader(train_list, batch_size = 256, shuffle = True)
        valid_loader = DataLoader(valid_list, batch_size = 256, shuffle = False)
        
        run_loss = model_finetuning(params = best_params, train_loader = train_loader, valid_loader = valid_loader,
                                pretrained_model_path = os.path.join(pretrained_model_directory, model_name), 
                                trained_model_path = os.path.join(model_directory, f'pcba_transfer_learning_model_fine_tuned_repeat_{repeat}_fold_{fold_no}.pt'))
        
        rmse, mae, mse, r2 = model_testing(params = best_params, test_loader = test_loader, 
                                           trained_model_path = os.path.join(model_directory, f'pcba_transfer_learning_model_fine_tuned_repeat_{repeat}_fold_{fold_no}.pt'))
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        mse_list.append(mse)
        r2_list.append(r2)
        print(f'Repetition no. {repeat+1}, Fold no. {fold_no + 1} has been completed! \n')
        
       
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

print('Model performance for best pretrained model with 0.1 lr:')
print(f'RMSE:{rmse_mean:.3f}Â±{rmse_sd:.3f}')
print(f'MAE:{mae_mean:.3f}Â±{mae_sd:.3f}')
print(f'MSE:{mse_mean:.3f}Â±{mse_sd:.3f}')
print(f'r2:{r2_mean:.3f}Â±{r2_sd:.3f}')
