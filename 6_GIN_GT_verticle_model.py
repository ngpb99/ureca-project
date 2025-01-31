from GNN_utils import engine, LipoFeatures
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import numpy as np
import optuna

'''
Function
'''
# Contains architecture of different combinations of GT and GIN models. Ensemble technique: vertical training (output embeddings from one model is passed sequentially into the other).
class VerticalModel_GIN_GT(torch.nn.Module):
    def __init__(self, hidden_size, num_layers_GIN, num_layers_GT, dropout, 
                 num_features, num_targets, edge_dim, heads):
        
        super(VerticalModel_GIN_GT, self).__init__()
        
        gin_layers = []
        for layers in range(num_layers_GIN):
            if layers == 0:
                gin_layers.append(GINConv(Sequential(
                    Linear(num_features, hidden_size), 
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
            else:
                gin_layers.append(GINConv(Sequential(
                    Linear(hidden_size, hidden_size),
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
        
        gt_layers = []
        for layers in range(num_layers_GT):
            if layers == 0:
                gt_layers.append(TransformerConv(in_channels = hidden_size, 
                                                out_channels = hidden_size, 
                                                heads = heads, 
                                                edge_dim = edge_dim, 
                                                dropout = dropout,
                                                beta = True))
            else:
                gt_layers.append(TransformerConv(in_channels = hidden_size * heads,
                                                out_channels = hidden_size,
                                                heads = heads,
                                                edge_dim = edge_dim, 
                                                dropout = dropout,
                                                beta = True))
        
        self.gin_model = nn.Sequential(*gin_layers)
        self.gt_model = nn.Sequential(*gt_layers)
        
        self.readout = Sequential(Linear(hidden_size * heads, hidden_size),
                                  ReLU(),
                                  Linear(hidden_size, num_targets))
        
    def forward(self, x, edge_attr, edge_index, batch):
        
        for layer_no, layers in enumerate(self.gin_model):
            if layer_no == 0:
                gin_embed = layers(x, edge_index)
            else:
                gin_embed = layers(gin_embed, edge_index)
        
        for layer_no, layers in enumerate(self.gt_model):
            if layer_no == 0:
                gt_embed = layers(gin_embed, edge_index, edge_attr)
            else:
                gt_embed = layers(gt_embed, edge_index, edge_attr)
        
        gt_output = global_mean_pool(gt_embed, batch)
        output = self.readout(gt_output)
        
        return output


class VerticalModel_GT_GIN(torch.nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, heads, 
                 dropout, edge_dim, num_layers_GIN, num_layers_GT):
        
        super(VerticalModel_GT_GIN, self).__init__()
        
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
        
        self.readout = Sequential(Linear(hidden_size, hidden_size),
                                  ReLU(),
                                  Linear(hidden_size, num_targets))
        
    def forward(self, x, edge_attr, edge_index, batch):
        
        for layer_no, layers in enumerate(self.gt_model):
            if layer_no == 0:
                gt_embed = layers(x, edge_index, edge_attr)
            else:
                gt_embed = layers(gt_embed, edge_index, edge_attr)
        
        for layer_no, layers in enumerate(self.gin_model):
            if layer_no == 0:
                gin_embed = layers(gt_embed, edge_index)
            else:
                gin_embed = layers(gin_embed, edge_index)
        
        gin_output = global_add_pool(gin_embed, batch)
        output = self.readout(gin_output)
        
        return output

# CombinedEmbeds: Output embeddings from both models are concatenated as final molecule representation.
class VerticalModel_GIN_GT_CombinedEmbeds(torch.nn.Module):
    def __init__(self, hidden_size, num_layers_GIN, num_layers_GT, dropout, 
                 num_features, num_targets, edge_dim, heads):
        
        super(VerticalModel_GIN_GT_CombinedEmbeds, self).__init__()
        
        gin_layers = []
        for layers in range(num_layers_GIN):
            if layers == 0:
                gin_layers.append(GINConv(Sequential(
                    Linear(num_features, hidden_size), 
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
            else:
                gin_layers.append(GINConv(Sequential(
                    Linear(hidden_size, hidden_size),
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
        
        gt_layers = []
        for layers in range(num_layers_GT):
            if layers == 0:
                gt_layers.append(TransformerConv(in_channels = hidden_size, 
                                                out_channels = hidden_size, 
                                                heads = heads, 
                                                edge_dim = edge_dim, 
                                                dropout = dropout,
                                                beta = True))
            else:
                gt_layers.append(TransformerConv(in_channels = hidden_size * heads,
                                                out_channels = hidden_size,
                                                heads = heads,
                                                edge_dim = edge_dim, 
                                                dropout = dropout,
                                                beta = True))
        
        self.gin_model = nn.Sequential(*gin_layers)
        self.gt_model = nn.Sequential(*gt_layers)
        
        self.readout = Sequential(Linear(hidden_size + hidden_size * heads, hidden_size),
                                  ReLU(),
                                  Linear(hidden_size, num_targets))
        
    def forward(self, x, edge_attr, edge_index, batch):
        
        all_embeds = []
        for layer_no, layers in enumerate(self.gin_model):
            if layer_no == 0:
                gin_embed = layers(x, edge_index)
            else:
                gin_embed = layers(gin_embed, edge_index)
        
        gin_final = global_add_pool(gin_embed, batch)
        all_embeds.append(gin_final)
        
        for layer_no, layers in enumerate(self.gt_model):
            if layer_no == 0:
                gt_embed = layers(gin_embed, edge_index, edge_attr)
            else:
                gt_embed = layers(gt_embed, edge_index, edge_attr)
        
        gt_final = global_mean_pool(gt_embed, batch)
        all_embeds.append(gt_final)
        
        concat_embeds = torch.concat(all_embeds, dim = 1)
        output = self.readout(concat_embeds)
        
        return output


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

def run_tuning(train_loader, valid_loader, params, model_type):
    if model_type == 'GIN_GT':
        model = VerticalModel_GIN_GT(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GT_GIN':
        model = VerticalModel_GT_GIN(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GT_GIN_Combined':
        model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GIN_GT_Combined':
        model = VerticalModel_GIN_GT_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    else:
        print('model type not specified')
        return
        
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = DEVICE)
    
    best_loss = np.inf
    early_stopping_counter = 0
    
    for epochs in range(EPOCHS):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        print(f'Epoch: {epochs+1}/{EPOCHS}, train_loss: {train_loss}, valid_loss: {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
        
        else:
            early_stopping_counter += 1
            print(f'Early stop counter: {early_stopping_counter}')
        
        if early_stopping_counter > PATIENCE:
            print('Commencing early stopping')
            break
        
    return best_loss


def run_training(train_loader, valid_loader, params, trained_model_path, model_type):
    if model_type == 'GIN_GT':
        model = VerticalModel_GIN_GT(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GT_GIN':
        model = VerticalModel_GT_GIN(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GT_GIN_Combined':
        model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GIN_GT_Combined':
        model = VerticalModel_GIN_GT_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
        
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = DEVICE)
    
    best_loss = np.inf
    early_stopping_counter = 0
    
    for epochs in range(EPOCH):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        print(f'For {epochs+1}/{EPOCH}, train loss is {train_loss}, valid loss is {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), trained_model_path)
            print('saving model...')
        
        else:
            early_stopping_counter += 1
            print(f'early stopping counter: {early_stopping_counter}')
        
        if early_stopping_counter > PATIENCE:
            print('Commencing early stopping')
            break


def run_testing(test_loader, params, trained_model_path, model_type):
    if model_type == 'GIN_GT':
        model = VerticalModel_GIN_GT(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GT_GIN':
        model = VerticalModel_GT_GIN(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GT_GIN_Combined':
        model = VerticalModel_GT_GIN_CombinedEmbeds(hidden_size = params['hidden_size'], 
                                     num_layers_GIN = params['num_layers_GIN'], 
                                     num_layers_GT = params['num_layers_GT'],
                                     dropout = params['dropout'],
                                     heads = params['heads'],
                                     num_features = NUM_FEATURES, num_targets = NUM_TARGETS, edge_dim = EDGE_DIM)
    elif model_type == 'GIN_GT_Combined':
        model = VerticalModel_GIN_GT_CombinedEmbeds(hidden_size = params['hidden_size'], 
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
Model Tuning and Testing
'''
NUM_FEATURES = 30
NUM_TARGETS = 1
EDGE_DIM = 11
DEVICE = 'cuda'
EPOCHS = 300
PATIENCE = 10
N_SPLITS = 5

def objective(trial):
    params = {'hidden_size': trial.suggest_int('hidden_size', 64, 512),
              'num_layers_GIN': trial.suggest_int('num_layers_GIN', 1, 3),
              'num_layers_GT': trial.suggest_int('num_layers_GT', 1, 3),
              'dropout': trial.suggest_float('dropout', 0.1, 0.4),
              'heads': trial.suggest_int('heads', 5, 10),
              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 9e-3, log=True)}
    
    dataset = LipoFeatures(root = './Processed data/graph data/lipo_train',
                           filename = 'raw_data_train.csv')
    kf = KFold(n_splits = N_SPLITS)
    final_loss = 0
    
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        print(f'Fold No.:{fold_no}')
        
        train_list = []
        valid_list = []
        for t in train_idx:
            train_list.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{t}.pt'))
        for v in valid_idx:
            valid_list.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{v}.pt'))
        
        train_loader = DataLoader(train_list, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_list, batch_size=256, shuffle=False)
        
        train_loss = run_tuning(train_loader = train_loader, valid_loader = valid_loader, params = params, model_type = model_type)
        final_loss += train_loss
        
    return(final_loss/N_SPLITS)

model_type = 'GT_GIN_Combined'

''' 
Choose:
1. 'GIN_GT'
2. 'GT_GIN'
3. 'GT_GIN_Combined'
4. 'GIN_GT_Combined'
'''

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials=30)
trial_ = study.best_trial
print(f'best trial:{trial_.values}')
print(f'Best parameters ({model_type}): {trial_.params}')

# Best parameters (GIN_GT): {'hidden_size': 115, 'num_layers_GIN': 2, 'num_layers_GT': 1, 'dropout': 0.2602614704271828, 'heads': 8, 'learning_rate': 0.0017214191200265797}
# Best parameters (GT_GIN): {'hidden_size': 65, 'num_layers_GIN': 2, 'num_layers_GT': 3, 'dropout': 0.35412396215632164, 'heads': 9, 'learning_rate': 0.0010851690632437962}
# Best parameters (GT_GIN_Combined): {'hidden_size': 70, 'num_layers_GIN': 3, 'num_layers_GT': 2, 'dropout': 0.3350406411118792, 'heads': 10, 'learning_rate': 0.0010341410418530216}
# Best parameters (GIN_GT_Combined): {'hidden_size': 236, 'num_layers_GIN': 2, 'num_layers_GT': 1, 'dropout': 0.10950843924080078, 'heads': 10, 'learning_rate': 0.0010019566487394862}

model_directory = './Models/vertical_models'
best_params_GIN_GT = {'hidden_size': 115, 'num_layers_GIN': 2, 'num_layers_GT': 1, 'dropout': 0.2602614704271828, 'heads': 8, 'learning_rate': 0.0017214191200265797}
best_params_GT_GIN = {'hidden_size': 65, 'num_layers_GIN': 2, 'num_layers_GT': 3, 'dropout': 0.35412396215632164, 'heads': 9, 'learning_rate': 0.0010851690632437962}
best_params_GT_GIN_Combined = {'hidden_size': 70, 'num_layers_GIN': 3, 'num_layers_GT': 2, 'dropout': 0.3350406411118792, 'heads': 10, 'learning_rate': 0.0010341410418530216}
best_params_GIN_GT_Combined = {'hidden_size': 236, 'num_layers_GIN': 2, 'num_layers_GT': 1, 'dropout': 0.10950843924080078, 'heads': 10, 'learning_rate': 0.0010019566487394862}
N_REPETITION = 5
model_type = 'GIN_GT' #Change this!!!
'''
Args:
1. 'GIN_GT'
2. 'GT_GIN'
3. 'GT_GIN_Combined'
4. 'GIN_GT_Combined'
'''

data_train = LipoFeatures(root = 'C:/Users/Ng Ping Boon/Desktop/URECA/Processed data/graph data/lipo_train', 
                          filename = 'raw_data_train.csv')
data_test = LipoFeatures(root = 'C:/Users/Ng Ping Boon/Desktop/URECA/Processed data/graph data/lipo_test', 
                          filename = 'raw_data_test.csv')

test_list = []
for test_idx in range(len(data_test)):
    test_list.append(torch.load(f'C:/Users/Ng Ping Boon/Desktop/URECA/Processed data/graph data/lipo_test/processed/data_{test_idx}.pt'))

test_loader = DataLoader(test_list, batch_size = 256, shuffle = False)

kf = KFold(n_splits = N_SPLITS)

rmse_list = []
mae_list = []
mse_list = []
r2_list = []

for repeat in range(N_REPETITION):
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(data_train)):
        print(f'Commencing repetition {repeat+1}, Fold no. {fold_no + 1}')
        
        train_list = []
        valid_list = []
        for t_idx in train_idx:
            train_list.append(torch.load(f'C:/Users/Ng Ping Boon/Desktop/URECA/Processed data/graph data/lipo_train/processed/data_{t_idx}.pt'))
        for v_idx in valid_idx:
            valid_list.append(torch.load(f'C:/Users/Ng Ping Boon/Desktop/URECA/Processed data/graph data/lipo_train/processed/data_{v_idx}.pt'))
        
        train_loader = DataLoader(train_list, batch_size = 256, shuffle = True)
        valid_loader = DataLoader(valid_list, batch_size = 256, shuffle = False)
        
        if model_type == 'GIN_GT':
            run_training(train_loader = train_loader, valid_loader = valid_loader, params = best_params_GIN_GT, 
                       trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gin_gt', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                       model_type = model_type)
            rmse, mae, mse, r2 = run_testing(test_loader = test_loader, params = best_params_GIN_GT, 
                                             trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gin_gt', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                                             model_type = model_type)
            
        elif model_type == 'GT_GIN':
            run_training(train_loader = train_loader, valid_loader = valid_loader, params = best_params_GT_GIN, 
                       trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gt_gin', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                       model_type = model_type)
            rmse, mae, mse, r2 = run_testing(test_loader = test_loader, params = best_params_GT_GIN, 
                                             trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gt_gin', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                                             model_type = model_type)
            
        elif model_type == 'GIN_GT_Combined':
            run_training(train_loader = train_loader, valid_loader = valid_loader, params = best_params_GIN_GT_Combined, 
                       trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gin_gt_combined', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                       model_type = model_type)
            rmse, mae, mse, r2 = run_testing(test_loader = test_loader, params = best_params_GIN_GT_Combined, 
                                             trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gin_gt_combined', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                                             model_type = model_type)
            
        elif model_type == 'GT_GIN_Combined':
            run_training(train_loader = train_loader, valid_loader = valid_loader, params = best_params_GT_GIN_Combined, 
                       trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gt_gin_combined', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                       model_type = model_type)
            rmse, mae, mse, r2 = run_testing(test_loader = test_loader, params = best_params_GT_GIN_Combined, 
                                             trained_model_path = os.path.join(model_directory, 'trained_vertical_models_gt_gin_combined', f'vertical_model_repeat_{repeat}_fold_{fold_no}.pt'),
                                             model_type = model_type)
        
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
        
print(f'RMSE:{rmse_mean:.3f}±{rmse_sd:.3f}')
print(f'MAE:{mae_mean:.3f}±{mae_sd:.3f}')
print(f'MSE:{mse_mean:.3f}±{mse_sd:.3f}')
print(f'r2:{r2_mean:.3f}±{r2_sd:.3f}')

# GIN_GT
# RMSE:0.859±0.096
# MAE:0.661±0.078
# MSE:0.750±0.174
# r2:0.766±0.054

# GT_GIN
# RMSE:0.669±0.042
# MAE:0.518±0.038
# MSE:0.451±0.058
# r2:0.858±0.018

# GT_GIN_Combined
# RMSE:0.648±0.026
# MAE:0.499±0.021
# MSE:0.422±0.034
# r2:0.867±0.010

# GIN_GT_Combined
# RMSE:0.670±0.032
# MAE:0.517±0.026
# MSE:0.451±0.043
# r2:0.858±0.013
