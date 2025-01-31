import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from GNN_utils import engine, LipoFeatures

'''
Function
'''
class ParallelModel(torch.nn.Module):
    def __init__(
            self, hidden_size, num_layers_GIN, num_targets, 
            num_features, num_layers_GT, heads, dropout, edge_dim
                 ):
        
        super(ParallelModel, self).__init__()
        gin_layers = []
        gt_layers = []
        for layers in range(num_layers_GIN):
            if len(gin_layers) == 0:
                gin_layers.append(
                    GINConv(Sequential(
                    Linear(num_features, hidden_size),
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
                
            elif len(gin_layers) > 0:
                gin_layers.append(
                    GINConv(Sequential(
                    Linear(hidden_size, hidden_size),
                    ReLU(),
                    Linear(hidden_size, hidden_size)
                    )))
        
        for layers in range(num_layers_GT):
            if len(gt_layers) == 0:
                gt_layers.append(
                    TransformerConv(
                        in_channels = num_features,
                        out_channels = hidden_size,
                        heads = heads,
                        dropout = dropout,
                        edge_dim = edge_dim,
                        beta = True
                        ))
            
            elif len(gt_layers) > 0:
                gt_layers.append(
                    TransformerConv(
                        in_channels = hidden_size * heads,
                        out_channels = hidden_size,
                        heads = heads,
                        dropout = dropout,
                        edge_dim = edge_dim,
                        beta = True
                        ))
        
        self.gin_model = nn.Sequential(*gin_layers)
        self.gt_model = nn.Sequential(*gt_layers)
        
        self.readout = Sequential(
                        Linear(hidden_size + hidden_size * heads, hidden_size),
                        ReLU(),
                        Linear(hidden_size, num_targets))
        
    def forward(self, x, edge_attr, edge_index, batch):
        all_embeddings = []
        
        for layer_no, layer in enumerate(self.gin_model):
            if layer_no == 0:
                gin_emb = layer(x, edge_index)
            else:
                gin_emb = layer(gin_emb, edge_index)
                
        gin_emb = global_add_pool(gin_emb, batch)
        all_embeddings.append(gin_emb)
        
        for layer_no, layer in enumerate(self.gt_model):
            if layer_no == 0:
                gt_emb = layer(x, edge_index, edge_attr)
            else:
                gt_emb = layer(gt_emb, edge_index, edge_attr)
        
        gt_emb = global_mean_pool(gt_emb, batch)
        all_embeddings.append(gt_emb)
        concat_emb = torch.concat(all_embeddings, dim=1)
        
        output = self.readout(concat_emb)
        return output


def run_tuning(train_loader, valid_loader, params):
    model = ParallelModel(hidden_size = params['hidden_size'], num_layers_GIN = params['num_layers_GIN'],
                          num_targets = NUM_TARGETS, num_features = NUM_FEATURES, 
                          num_layers_GT = params['num_layers_GT'], heads = params['heads'], dropout = params['dropout'], 
                          edge_dim = EDGE_DIM)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    eng = engine(model = model, device = DEVICE, optimizer = optimizer)
    
    best_loss = np.inf
    early_stopping_counter = 0
    
    for epochs in range(EPOCHS):
        train_loss = eng.train(train_loader)
        valid_loss = eng.validate(valid_loader)
        print(f'For Epoch {epochs+1}, Train loss is {train_loss}, Valid loss is {valid_loss}')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
        
        else:
            early_stopping_counter += 1
            print(f'Early stopping counter: {early_stopping_counter}')
        
        if early_stopping_counter > PATIENCE:
            print('Commencing early stopping')
            break
    
    return best_loss


def run_training(train_loader, valid_loader, params, trained_model_path):
    model = ParallelModel(hidden_size = params['hidden_size'], num_layers_GIN = params['num_layers_GIN'], 
                          num_targets = NUM_TARGETS, num_features = NUM_FEATURES, num_layers_GT = params['num_layers_GT'],
                          heads = params['heads'], dropout = params['dropout'], edge_dim = EDGE_DIM)
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

def run_testing(test_loader, params, trained_model_path):
    model = ParallelModel(hidden_size = params['hidden_size'], num_layers_GIN = params['num_layers_GIN'], 
                          num_targets = NUM_TARGETS, num_features = NUM_FEATURES, num_layers_GT = params['num_layers_GT'],
                          heads = params['heads'], dropout = params['dropout'], edge_dim = EDGE_DIM)
    model.load_state_dict(torch.load(trained_model_path))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = DEVICE)
    
    rmse, mae, mse, r2 = eng.test(test_loader)
    
    return rmse, mae, mse, r2

'''
Model Tuning and Testing
'''
NUM_TARGETS = 1
NUM_FEATURES = 30
EDGE_DIM = 11
DEVICE = 'cuda'
EPOCHS = 300
PATIENCE = 10
N_SPLITS = 5

def objective(trial):
    
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 64, 512),
        'num_layers_GIN': trial.suggest_int('num_layers_GIN', 1, 3),
        'num_layers_GT': trial.suggest_int('num_layers_GT', 1, 3),
        'heads': trial.suggest_int('heads', 5, 10),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 9e-3, log=True)
              }
    
    print('Hyperparameter testing for current run',
          '\nhidden size:', params['hidden_size'], 
          '\nnum_layers_GIN:', params['num_layers_GIN'], 
          '\nnum_layers_GT:', params['num_layers_GT'], 
          '\nheads:', params['heads'], 
          '\ndropout:', params['dropout'], 
          '\nlearning rate:', params['learning_rate'], '\n')
    
    dataset_cv = LipoFeatures(root = './Processed data/graph data/lipo_train', 
                              filename = 'raw_data_train.csv')
    
    kf = KFold(n_splits=N_SPLITS)
    total_loss = 0
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(dataset_cv)):
        print(f'Fold no. {fold_no + 1} / 5')
        train_list = []
        valid_list = []
        for t_idx in train_idx:
            train_list.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{t_idx}.pt'))
        for v_idx in valid_idx:
            valid_list.append(torch.load(f'./Processed data/graph data/lipo_train/processed/data_{v_idx}.pt'))
        
        train_loader = DataLoader(train_list, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_list, batch_size=256, shuffle=False)
        
        loss = run_tuning(train_loader = train_loader, valid_loader = valid_loader, params = params)
        total_loss += loss
    
    return (total_loss/N_SPLITS)

study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 30)
trial_ = study.best_trial
best_params = trial_.params
print(f'best trial:{trial_.values}')
print(f'Best parameters: {trial_.params}')

# best trial:[0.469861992767879]
# Best parameters: {'hidden_size': 197, 'num_layers_GIN': 3, 'num_layers_GT': 3, 'heads': 7, 'dropout': 0.30817392636332563, 'learning_rate': 0.0010816202178616917}

N_REPETITION = 5
model_directory = './Models/parallel_model'
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
        
        run_training(train_loader = train_loader, valid_loader = valid_loader, params = best_params, 
                   trained_model_path = os.path.join(model_directory, f'parallel_model_repeat_{repeat}_fold_{fold_no}.pt'))
        rmse, mae, mse, r2 = run_testing(test_loader = test_loader, params = best_params, 
                                         trained_model_path = os.path.join(model_directory, f'parallel_model_repeat_{repeat}_fold_{fold_no}.pt'))
        
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

# RMSE:0.692±0.047
# MAE:0.537±0.046
# MSE:0.482±0.068
# r2:0.848±0.021
