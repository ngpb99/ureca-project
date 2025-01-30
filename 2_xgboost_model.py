import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

'''
Functions
'''
def run_tuning(descriptor_type, data, params, seed): 
    if descriptor_type == 'mol_desc':
        x_data = data.iloc[:,:40]
        
    elif descriptor_type == 'morgan_fp':
        x_data = data.iloc[:,:1024]
        
    elif descriptor_type == 'rdkit_fp':
        x_data = data.iloc[:,:1024]
        
    elif descriptor_type == 'maccs_keys':
        x_data = data.iloc[:,:167]
    
    elif descriptor_type == 'ecfp':
        x_data = data.iloc[:,:2047]
        
    y_data = data['LogD']
    
    kf = KFold(n_splits = 5)
    xgb_model = XGBRegressor(learning_rate=params['learning_rate'], max_depth=params['max_depth'],
                             gamma=params['gamma'], alpha=params['alpha'], subsample = params['subsample'], reg_lambda=params['reg_lambda'],
                             min_child_weight=params['min_child_weight'], colsample_bytree = params['colsample_bytree'],
                             n_estimators = params['n_estimators'], seed = seed, objective='reg:squarederror',
                             early_stopping_rounds = 10, device = 'cuda')
    # early stopping is set to 10 here, indicating that if after the creation of 10 subsequent trees and rmse does not get better, then we stop making additional trees.
    
    total_rmse = 0
    
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(x_data)):
        x_train_fold, x_valid_fold = x_data.iloc[train_idx], x_data.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_data.iloc[train_idx], y_data.iloc[valid_idx]
        
        xgb_model.fit(x_train_fold, y_train_fold.to_numpy().reshape(-1), 
                      eval_set = [(x_valid_fold, y_valid_fold)])
        # eval_set is used for early stopping for the model to use the validation data to evaluate the performance.
        # this part here is similar to y_pred = xgb_model.predict(x_valid_fold) but instead of keeping track of the average RMSE for a particular set of hyperparameters, it is for the model to evaluate when to stop making additional trees.
        
        y_pred = xgb_model.predict(x_valid_fold)
        mse = mean_squared_error(y_valid_fold, y_pred)
        
        rmse = np.sqrt(mse)
        total_rmse = total_rmse + rmse
        print(f'Fold: {fold_no}, RMSE: {rmse}')
        
    return total_rmse/5

def train_model(descriptor_type, train_data, best_params, n_repititions, trained_model_directory, seed):
    if descriptor_type == 'mol_desc':
        x_data = train_data.iloc[:,:40]
        
    elif descriptor_type == 'morgan_fp':
        x_data = train_data.iloc[:,:1024]
        
    elif descriptor_type == 'rdkit_fp':
        x_data = train_data.iloc[:,:1024]
        
    elif descriptor_type == 'maccs_keys':
        x_data = train_data.iloc[:,:167]
    
    elif descriptor_type == 'ecfp':
        x_data = train_data.iloc[:,:2047]
        
    y_data = train_data['LogD']
    
    kf = KFold(n_splits = 5)
    xgb_model = XGBRegressor(learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'],
                             gamma=best_params['gamma'], alpha=best_params['alpha'], subsample = best_params['subsample'], reg_lambda=best_params['reg_lambda'],
                             min_child_weight=best_params['min_child_weight'], colsample_bytree = best_params['colsample_bytree'],
                             n_estimators = best_params['n_estimators'], seed = seed, objective='reg:squarederror',
                             early_stopping_rounds = 10, device = 'cuda')
    
    for i in range(n_repititions):
        for fold_no, (train_idx, test_idx) in enumerate(kf.split(x_data)):
            x_train_fold, y_train_fold = x_data.iloc[train_idx], y_data.iloc[train_idx]
            x_valid_fold, y_valid_fold = x_data.iloc[test_idx], y_data.iloc[test_idx]
            xgb_model.fit(x_train_fold, y_train_fold.to_numpy().reshape(-1), eval_set = [(x_valid_fold, y_valid_fold)])
            joblib.dump(xgb_model, os.path.join(trained_model_directory, f'trained_xgb_model_{descriptor_type}_repeat_{i}_fold_{fold_no}_ver2.pkl'))
    print('training completed')
    
def test_model(descriptor_type, test_data, n_repititions, trained_model_directory):
    if descriptor_type == 'mol_desc':
        x_data = test_data.iloc[:,:40]
        
    elif descriptor_type == 'morgan_fp':
        x_data = test_data.iloc[:,:1024]
        
    elif descriptor_type == 'rdkit_fp':
        x_data = test_data.iloc[:,:1024]
        
    elif descriptor_type == 'maccs_keys':
        x_data = test_data.iloc[:,:167]
        
    elif descriptor_type == 'ecfp':
        x_data = test_data.iloc[:,:2047]
        
    y_data = test_data['LogD']
    rmse_total = []
    mse_total = []
    mae_total = []
    r2_total = []
    
    kf = KFold(n_splits = 5)
    
    for i in range(n_repititions):
        for fold_no, (train_idx, test_idx) in enumerate(kf.split(x_data)):
            rf_model = joblib.load(os.path.join(trained_model_directory, f'trained_xgb_model_{descriptor_type}_repeat_{i}_fold_{fold_no}_ver2.pkl'))
            y_pred = rf_model.predict(x_data)
            
            mse = mean_squared_error(y_data, y_pred)
            mse_total.append(mse)
            
            rmse = np.sqrt(mse)
            rmse_total.append(rmse)
            
            r2 = r2_score(y_data, y_pred)
            r2_total.append(r2)
            
            mae = mean_absolute_error(y_data, y_pred)
            mae_total.append(mae)
    return np.mean(mse_total), np.std(mse_total), np.mean(rmse_total), np.std(rmse_total), np.mean(r2_total), np.std(r2_total), np.mean(mae_total), np.std(mae_total),

'''
XGBoost Model tuning
'''
mol_desc_train = pd.read_csv('./Processed data/Data splitting/mol_desc_train.csv')
morgan_fp_train = pd.read_csv('./Processed data/Data splitting/morgan_fp_train.csv')
maccs_keys_train = pd.read_csv('./Processed data/Data splitting/maccs_keys_train.csv')
rdkit_fp_train = pd.read_csv('./Processed data/Data splitting/rdkit_fp_train.csv')
ecfp_train = pd.read_csv('./Processed data/Data splitting/ecfp_train.csv')

descriptor_type = ['mol_desc', 'morgan_fp', 'rdkit_fp', 'maccs_keys', 'ecfp']
data = [mol_desc_train, morgan_fp_train, rdkit_fp_train, maccs_keys_train, ecfp_train]
seed = 40
for i in range(len(descriptor_type)):
  data_1 = data[i]
  descriptor_type_1 = descriptor_type[i]
  
  def objective(trial):
      params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 1, 15),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                'n_estimators': 100000,
                'alpha': trial.suggest_float('alpha', 0.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0)}
    rmse = run_tuning(descriptor_type, data, params, seed)
    return rmse

  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=100)
  trial_ = study.best_trial
  print('best trial:', trial_.values)
  print(f'Best parameters: {trial_.params}')

'''
Model training and testing
'''
best_params_moldesc = {'learning_rate': 0.07942662344309474, 
                  'max_depth': 9, 'gamma': 0.005555904741154288, 
                  'min_child_weight': 4, 'colsample_bytree': 0.7069111063299118, 
                  'alpha': 5.222957925177122, 'subsample': 0.5535940327022462, 
                  'reg_lambda': 7.731936454170216, 'n_estimators': 100000}

best_params_morganfp = {'learning_rate': 0.11828305885246676, 
                  'max_depth': 10, 'gamma': 0.002382983604344406, 
                  'min_child_weight': 5, 'colsample_bytree': 0.957063841248394, 
                  'alpha': 2.1347245763122213, 'subsample': 0.8889365243492054, 
                  'reg_lambda': 9.813849583870338, 'n_estimators': 100000}

best_params_rdkitfp = {'learning_rate': 0.20316787534076194, 
                  'max_depth': 9, 'gamma': 0.03614576176901896, 
                  'min_child_weight': 0, 'colsample_bytree': 0.5119293105901023, 
                  'alpha': 1.021970729964298, 'subsample': 0.7876298812694362, 
                  'reg_lambda': 5.14125031469706, 'n_estimators': 100000}

best_params_maccs = {'learning_rate': 0.10656971966624768, 
                  'max_depth': 9, 'gamma': 0.013852257271072126, 
                  'min_child_weight': 10, 'colsample_bytree': 0.8820969243676807, 
                  'alpha': 0.20016119402464838, 'subsample': 0.8980594352839236, 
                  'reg_lambda': 4.194346666525757, 'n_estimators': 100000}

best_params_ecfp = {'learning_rate': 0.17927599893158658, 
                  'max_depth': 7, 'gamma': 0.013418017057222748, 
                  'min_child_weight': 6, 'colsample_bytree': 0.6899216637393378, 
                  'alpha': 0.45442336002443284, 'subsample': 0.79198007558107, 
                  'reg_lambda': 9.42911742276986, 'n_estimators': 100000}

#mol_desc_test_train
mol_desc_train = pd.read_csv('./Processed data/Data splitting/mol_desc_train.csv')
mol_desc_test = pd.read_csv('./Processed data/Data splitting/mol_desc_test.csv')
#morgan_fp_test_train
morgan_fp_train = pd.read_csv('./Processed data/Data splitting/morgan_fp_train.csv')
morgan_fp_test = pd.read_csv('./Processed data/Data splitting/morgan_fp_test.csv')
#maccs_keys_test_train
maccs_keys_train = pd.read_csv('./Processed data/Data splitting/maccs_keys_train.csv')
maccs_keys_test = pd.read_csv('./Processed data/Data splitting/maccs_keys_test.csv')
#rdkit_fp_test_train
rdkit_fp_train = pd.read_csv('./Processed data/Data splitting/rdkit_fp_train.csv')
rdkit_fp_test = pd.read_csv('./Processed data/Data splitting/rdkit_fp_test.csv')
#ecfp_test_train
ecfp_train = pd.read_csv('./Processed data/Data splitting/ecfp_train.csv')
ecfp_test = pd.read_csv('./Processed data/Data splitting/ecfp_test.csv')

descriptor_type = ['mol_desc', 'morgan_fp', 'rdkit_fp', 'maccs_keys', 'ecfp']
train_data = [mol_desc_train, morgan_fp_train, maccs_keys_train, rdkit_fp_train, ecfp_train]
test_data = [mol_desc_test, morgan_fp_test, maccs_keys_test, rdkit_fp_test, ecfp_test]
best_params = [best_params_moldesc, best_params_morganfp, best_params_maccs, best_params_rdkitfp, best_params_ecfp]
trained_model_directory = './Models/xgb_models'
n_repititions = 5
seed = 40

for i in range(len(descriptor_type)):
  train_model(descriptor_type[i], train_data[i], best_params[i], n_repititions, trained_model_directory, seed)
  mse_mean, mse_std, rmse_mean, rmse_std, r2_mean, r2_std, mae_mean, mae_std = test_model(descriptor_type[i], test_data[i], n_repititions, trained_model_directory)
  print(f'{descriptor_type[i]} rf model results')
  print(f'Avg MSE: {mse_mean} ± {mse_std}')
  print(f'Avg RMSE: {rmse_mean} ± {rmse_std}')
  print(f'Avg r2 score: {r2_mean} ± {r2_std}')
  print(f'Avg MAE: {mae_mean} ± {mae_std}')

# mol_desc xgb model results
# Avg MSE: 0.6848496952405589 ± 0.020484649788770702
# Avg RMSE: 0.8274644343047014 ± 0.012341199348507028
# Avg r2 score: 0.8228794077594839 ± 0.005297882626861883
# Avg MAE: 0.55920163589917 ± 0.009058443655850096

# morgan_fp xgb model results
# Avg MSE: 0.8338433753817511 ± 0.04211760660236695
# Avg RMSE: 0.9128586780686806 ± 0.02307399524252272
# Avg r2 score: 0.7612772573822271 ± 0.012057936607112
# Avg MAE: 0.6274338970588467 ± 0.015544438829575583

# rdkit_fp xgb model results
# Avg MSE: 1.465334053612647 ± 0.014236775777377761
# Avg RMSE: 1.2104955890143996 ± 0.0058721877804630155
# Avg r2 score: 0.5804864864825723 ± 0.0040758759497896155
# Avg MAE: 0.8429021749481674 ± 0.007783721474292698

# maccs_keys xgb model results
# Avg MSE: 0.9168726730946113 ± 0.034846575582979815
# Avg RMSE: 0.9573611504591488 ± 0.018229116438986972
# Avg r2 score: 0.737506628205534 ± 0.00997629811497629
# Avg MAE: 0.679573175983281 ± 0.02309642031759074

# ecfp xgb model results
# Avg MSE: 0.44355693488305087 ± 0.026218557978647625
# Avg RMSE: 0.6657109978389443 ± 0.019641849183002054
# Avg r2 score: 0.8730131687453444 ± 0.00750616512996862
# Avg MAE: 0.4704606252640718 ± 0.0148688391797824
