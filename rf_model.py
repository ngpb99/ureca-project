import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import optuna
import os
import joblib

'''
Functions
'''
def canonical_smiles(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    return smiles

def data_cleaning():
    lipo_data1 = pd.read_csv('./Raw data/Lipophilicity.csv')
    lipo_data1 = lipo_data1.rename(columns = {'CMPD_CHEMBLID':'ID', 'exp':'LogD'})
    lipo_data1_cleaned = lipo_data1.dropna(subset=['LogD']) #drop entry with no data
    
    lipo_data2 = pd.read_csv('./Raw data/Lipophilicity 2.csv', delimiter= ';')
    cols = ['ChEMBL ID', 'Smiles', 'CX LogD']
    lipo_data2 = lipo_data2[cols] #selecting the cols of interest, while discarding the rest
    lipo_data2_cleaned = lipo_data2.dropna(subset=['CX LogD'])
    lipo_data2_cleaned = lipo_data2_cleaned.reset_index(drop=True)
    lipo_data2_cleaned = lipo_data2_cleaned.rename(columns = {'ChEMBL ID':'ID', 'Smiles':'smiles', 'CX LogD':'LogD'})
    
    merge_lipo = pd.concat([lipo_data1_cleaned, lipo_data2_cleaned], axis = 0) #combine both dataframes together
    Canon_smiles = canonical_smiles(merge_lipo['smiles']) #identify molecules that are represented differently in SMILES fromat, but are the supposedly the same moecule.
    merge_lipo['smiles'] = Canon_smiles 
    duplicate_smiles = merge_lipo[merge_lipo['smiles'].duplicated(keep = False)] #record all duplicates with same smiles
    #Duplicated ID can represent different molecules, thus it is not a concern
    
    average_lipo_duplicates = duplicate_smiles.copy()
    average_lipo_duplicates['Avg LogD'] = average_lipo_duplicates.groupby('smiles')['LogD'].transform('mean') #create a new col which record the mean LogD value of duplicated smiles
    average_lipo_duplicates = average_lipo_duplicates[average_lipo_duplicates['smiles'].duplicated(keep='first')] #drop the duplicated entry
    average_lipo_duplicates = average_lipo_duplicates.drop(columns = 'LogD') #drop LogD and subsequently replace with Avg LogD
    average_lipo_duplicates = average_lipo_duplicates.rename(columns = {'Avg LogD':'LogD'})
    
    merge_lipo_cleaned = merge_lipo.drop_duplicates(subset='smiles', keep=False) #drop all duplicates as thier data has been prepared in average_lipo_duplicates
    merge_lipo_cleaned = pd.concat([merge_lipo_cleaned, average_lipo_duplicates], axis = 0) #merge the prepared dataframe to original dataframe
    merge_lipo_cleaned = merge_lipo_cleaned.reset_index(drop=True) #reset index
    merge_lipo_cleaned.to_csv('./Processed data/cleaned_raw_data.csv', index = False)

def Mol_descriptors(data):
    df_smiles = data['smiles']
    all_descs = [x[0] for x in Descriptors._descList if x[0] is not None]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descs)
    
    desc_list = []
    for smile in df_smiles:
        mol = Chem.MolFromSmiles(smile)
        desc = calc.CalcDescriptors(mol) #calculate descriptors for each molecule
        desc_list.append(desc)
    
    df_desc = pd.DataFrame(desc_list, columns=all_descs)
    var = pd.DataFrame(df_desc.var())
    no_var_desc = []
    for index, row in var.iterrows():
        if row[0] == 0:
            no_var_desc.append(index)
    df_desc.drop(columns=no_var_desc, inplace=True) #drop all descriptors with no variance
    df_desc['LogD'] = data['LogD']
    df_desc.dropna(inplace=True, axis=0)
    df_desc.to_csv('./Processed data/mol_desc.csv', index = False)

def morgan_fp(data, rad, bit):
    Morgan_fpts = []
    for i in data['smiles']:
        mol = Chem.MolFromSmiles(i) 
        fpts =  AllChem.GetMorganFingerprintAsBitVect(mol, rad, bit) #radius = 2, bit size = 1024
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)  
    morgan_array = np.array(Morgan_fpts)
    morgan_fp = pd.DataFrame(morgan_array)
    morgan_fp = pd.concat([morgan_fp, target_variables], axis = 1)
    morgan_fp.to_csv('./Processed data/morgan_fp.csv', index = False)

def maccs_keys(data):
    maccs_keys = []
    for i in data['smiles']:
        mol = Chem.MolFromSmiles(i)
        keys = MACCSkeys.GenMACCSKeys(mol)
        mkeys = np.array(keys)
        maccs_keys.append(mkeys)
    maccs_array = np.array(maccs_keys)
    maccs_keys = pd.DataFrame(maccs_array)
    maccs_keys = pd.concat([maccs_keys, target_variables], axis = 1)
    maccs_keys.to_csv('./Processed data/maccs_keys.csv', index = False)

def rdkitfp(data, minpath, maxpath, bit):
    rdkitfp = []
    for i in data['smiles']:
        mol = Chem.MolFromSmiles(i)
        fp = Chem.rdmolops.RDKFingerprint(mol, minPath=minpath, maxPath=maxpath, fpSize = bit)
        rdfp = np.array(fp)
        rdkitfp.append(rdfp)
    rd_array = np.array(rdkitfp)
    rdkit = pd.DataFrame(rd_array)
    rdkit = pd.concat([rdkit, target_variables], axis = 1)
    rdkit.to_csv('./Processed data/rdkit_fp.csv', index = False)

def ECFP4(data, rad):
    ecfp = []
    invgen = AllChem.GetMorganFeatureAtomInvGen()
    for i in data['smiles']:
        mol = Chem.MolFromSmiles(i)
        ffpgen = AllChem.GetMorganGenerator(radius=rad, atomInvariantsGenerator=invgen)
        ffp = ffpgen.GetCountFingerprintAsNumPy(mol)
        ecfp.append(ffp)
    ecfp_array = np.array(ecfp)
    ecfp_df = pd.DataFrame(ecfp_array)
    ecfp_df = pd.concat([ecfp_df, target_variables], axis = 1)
    ecfp_df.to_csv('./Processed data/ecfp_df.csv', index = False)

def split_data(descriptor_type, split_fraction, seed):
    if descriptor_type == 'mol_desc':
        mol_desc_test = mol_desc.sample(frac = split_fraction, random_state = seed)
        mol_desc_train = mol_desc.drop(mol_desc_test.index)
        mol_desc_test.to_csv('./Processed data/Data splitting/mol_desc_test.csv', index = False)
        mol_desc_train.to_csv('./Processed data/Data splitting/mol_desc_train.csv', index = False)
    
    elif descriptor_type == 'morgan_fp':
        morgan_fp_test = morgan_fp.sample(frac = split_fraction, random_state = seed)
        morgan_fp_train = morgan_fp.drop(morgan_fp_test.index)
        morgan_fp_test.to_csv('./Processed data/Data splitting/morgan_fp_test.csv', index = False)
        morgan_fp_train.to_csv('./Processed data/Data splitting/morgan_fp_train.csv', index = False)
    
    elif descriptor_type == 'maccs_keys':
        maccs_keys_test = maccs_keys.sample(frac = split_fraction, random_state = seed)
        maccs_keys_train = maccs_keys.drop(maccs_keys_test.index)
        maccs_keys_test.to_csv('./Processed data/Data splitting/maccs_keys_test.csv', index = False)
        maccs_keys_train.to_csv('./Processed data/Data splitting/maccs_keys_train.csv', index = False)
    
    elif descriptor_type == 'rdkit_fp':
        rdkit_fp_test = rdkit_fp.sample(frac = split_fraction, random_state = seed)
        rdkit_fp_train = rdkit_fp.drop(rdkit_fp_test.index)
        rdkit_fp_test.to_csv('./Processed data/Data splitting/rdkit_fp_test.csv', index = False)
        rdkit_fp_train.to_csv('./Processed data/Data splitting/rdkit_fp_train.csv', index = False)
    
    elif descriptor_type == 'raw':
        raw_data_test = raw_data.sample(frac = split_fraction, random_state = seed)
        raw_data_train = raw_data.drop(raw_data_test.index)
        raw_data_test.to_csv('./Processed data/Data splitting/raw_data_test.csv', index = False)
        raw_data_train.to_csv('./Processed data/Data splitting/raw_data_train.csv', index = False)
    
    elif descriptor_type == 'ecfp':
        ecfp_data_test = ecfp.sample(frac = split_fraction, random_state = seed)
        ecfp_data_train = ecfp.drop(ecfp_data_test.index)
        ecfp_data_test.to_csv('./Processed data/Data splitting/ecfp_test.csv', index = False)
        ecfp_data_train.to_csv('./Processed data/Data splitting/ecfp_train.csv', index = False)

def run_tuning(descriptor_type, data, params, seed): 
    if descriptor_type == 'mol_desc':
        x_data = data.iloc[:,:40]
        
    elif descriptor_type == 'morgan_fp':
        x_data = data.iloc[:,:1024]
        
    elif descriptor_type == 'rdkit_fp':
        x_data = data.iloc[:,:1024]
        
    elif descriptor_type == 'maccs_keys':
        x_data = data.iloc[:,:167]
        
    y_data = data['LogD']
    
    kf = KFold(n_splits = 5)
    rf_model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state= seed)
    total_rmse = 0
    
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(x_data)):
        x_train_fold, x_valid_fold = x_data.iloc[train_idx], x_data.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_data.iloc[train_idx], y_data.iloc[valid_idx]
        
        rf_model.fit(x_train_fold, y_train_fold.to_numpy().reshape(-1))
        y_pred = rf_model.predict(x_valid_fold)
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
        
    y_data = train_data['LogD']
    
    kf = KFold(n_splits = 5)
    rf_model = RandomForestRegressor(n_estimators=best_params['n_estimator'], max_depth=best_params['max_depth'], random_state= seed)
    
    for i in range(n_repititions):
        for fold_no, (train_idx, test_idx) in enumerate(kf.split(x_data)):
            x_train_fold, y_train_fold = x_data.iloc[train_idx], y_data.iloc[train_idx]
            rf_model.fit(x_train_fold, y_train_fold.to_numpy().reshape(-1))
            joblib.dump(rf_model, os.path.join(trained_model_directory, f'trained_rf_model_{descriptor_type}_repeat_{i}_fold_{fold_no}_ver2.pkl'))
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
        
    y_data = test_data['LogD']
    rmse_total = []
    mse_total = []
    mae_total = []
    r2_total = []
    
    kf = KFold(n_splits = 5)
    
    for i in range(n_repititions):
        for fold_no, (train_idx, test_idx) in enumerate(kf.split(x_data)):
            rf_model = joblib.load(os.path.join(trained_model_directory, f'trained_rf_model_{descriptor_type}_repeat_{i}_fold_{fold_no}_ver2.pkl'))
            y_pred = rf_model.predict(x_data)
            
            mse = mean_squared_error(y_data, y_pred)
            mse_total.append(mse)
            
            rmse = np.sqrt(mse)
            rmse_total.append(rmse)
            
            r2 = r2_score(y_data, y_pred)
            r2_total.append(r2)
            
            mae = mean_absolute_error(y_data, y_pred)
            mae_total.append(mae)
    return np.mean(mse_total), np.std(mse_total), np.mean(rmse_total), np.std(rmse_total), np.mean(r2_total), np.std(r2_total), np.mean(mae_total), np.std(mae_total)

'''
Data Cleaning
'''
lipo_data1 = pd.read_csv('./Raw data/Lipophilicity.csv')
lipo_data1 = lipo_data1.rename(columns = {'CMPD_CHEMBLID':'ID', 'exp':'LogD'})
lipo_data1_cleaned = lipo_data1.dropna(subset=['LogD']) #drop entry with no data

lipo_data2 = pd.read_csv('./Raw data/Lipophilicity 2.csv', delimiter= ';')
cols = ['ChEMBL ID', 'Smiles', 'CX LogD']
lipo_data2 = lipo_data2[cols] #selecting the cols of interest, while discarding the rest
lipo_data2_cleaned = lipo_data2.dropna(subset=['CX LogD'])
lipo_data2_cleaned = lipo_data2_cleaned.reset_index(drop=True)
lipo_data2_cleaned = lipo_data2_cleaned.rename(columns = {'ChEMBL ID':'ID', 'Smiles':'smiles', 'CX LogD':'LogD'})

merge_lipo = pd.concat([lipo_data1_cleaned, lipo_data2_cleaned], axis = 0) #combine both dataframes together
Canon_smiles = canonical_smiles(merge_lipo['smiles']) #identify molecules that are represented differently in SMILES fromat, but are the supposedly the same moecule.
merge_lipo['smiles'] = Canon_smiles 
duplicate_smiles = merge_lipo[merge_lipo['smiles'].duplicated(keep = False)] #record all duplicates with same smiles
# Duplicated ID can represent different molecules, thus it is not a concern

average_lipo_duplicates = duplicate_smiles.copy()
average_lipo_duplicates['Avg LogD'] = average_lipo_duplicates.groupby('smiles')['LogD'].transform('mean') #create a new col which record the mean LogD value of duplicated smiles
average_lipo_duplicates = average_lipo_duplicates[average_lipo_duplicates['smiles'].duplicated(keep='first')] #drop the duplicated entry
average_lipo_duplicates = average_lipo_duplicates.drop(columns = 'LogD') #drop LogD and subsequently replace with Avg LogD
average_lipo_duplicates = average_lipo_duplicates.rename(columns = {'Avg LogD':'LogD'})

merge_lipo_cleaned = merge_lipo.drop_duplicates(subset='smiles', keep=False) #drop all duplicates as thier data has been prepared in average_lipo_duplicates
merge_lipo_cleaned = pd.concat([merge_lipo_cleaned, average_lipo_duplicates], axis = 0) #merge the prepared dataframe to original dataframe
merge_lipo_cleaned = merge_lipo_cleaned.reset_index(drop=True) #reset index

'''
Prcoessing and feature generation
'''
data_cleaning()
cleaned_data = pd.read_csv('./Processed data/cleaned_raw_data.csv')
Mol_descriptors(cleaned_data)
morgan_fp(cleaned_data, 2, 1024)
maccs_keys(cleaned_data)
rdkitfp(cleaned_data, 1, 2, 1024)
ECFP4(cleaned_data, 2)

'''
Data splitting
'''
raw_data = pd.read_csv('./Processed data/cleaned_raw_data.csv')
mol_desc = pd.read_csv('./Processed data/mol_desc.csv')
morgan_fp = pd.read_csv('./Processed data/morgan_fp.csv')
maccs_keys = pd.read_csv('./Processed data/maccs_keys.csv')
rdkit_fp = pd.read_csv('./Processed data/rdkit_fp.csv')
ecfp = pd.read_csv('./Processed data/ecfp_df.csv')

split_fraction = 0.2
seed = 40
split_data('mol_desc', split_fraction, seed)
split_data('morgan_fp', split_fraction, seed)
split_data('maccs_keys', split_fraction, seed)
split_data('rdkit_fp', split_fraction, seed)
split_data('ecfp', split_fraction, seed)

'''
RF Model tuning
'''
mol_desc_train = pd.read_csv('./Processed data/Data splitting/mol_desc_train.csv')
morgan_fp_train = pd.read_csv('./Processed data/Data splitting/morgan_fp_train.csv')
maccs_keys_train = pd.read_csv('./Processed data/Data splitting/maccs_keys_train.csv')
rdkit_fp_train = pd.read_csv('./Processed data/Data splitting/rdkit_fp_train.csv')

descriptor_type = ['mol_desc', 'morgan_fp', 'rdkit_fp', 'maccs_keys']
data = [mol_desc_train, morgan_fp_train, rdkit_fp_train, maccs_keys_train]
seed = 40

for i in range(len(descriptor_type)):
  data_1 = data[i]
  descriptor_type_1 = descriptor_type[i]
  
  def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 300),
            'max_depth': trial.suggest_int('max_depth', 10, 100)
            }
    rmse = run_tuning(descriptor_type_1, data_1, params, seed)
    return rmse
      
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=40)
  trial_ = study.best_trial
  print('best trial:', trial_.values)
  print(f'Best parameters: {trial_.params}')

# mol_desc, n_estimators 100-200, max_depth 10-100, n_trials 40. Best parameters: {'n_estimators': 234, 'max_depth': 21}
# Morgan_fp, n_estimators 100-200, max_depth 10-100, n_trials 40. Best parameters: {'n_estimators': 199, 'max_depth': 98}
# rdkit_fp, n_estimators 100-200, max_depth 10-100, n_trials 40. Best parameters: {'n_estimators': 180, 'max_depth': 27}
# maccs_keys, n_estimators 100-200, max_depth 10-100, n_trials 40. Best parameters: {'n_estimator': 192, 'max_depth': 76}

'''
Model training and testing
'''
best_params_moldesc = {'n_estimator': 234, 'max_depth': 21}
best_params_morganfp = {'n_estimator': 199, 'max_depth': 98}
best_params_rdkitfp = {'n_estimator': 180, 'max_depth': 27}
best_params_maccs = {'n_estimator': 192, 'max_depth': 76}

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

descriptor_type = ['mol_desc', 'morgan_fp', 'rdkit_fp', 'maccs_keys']
train_data = [mol_desc_train, morgan_fp_train, maccs_keys_train, rdkit_fp_train]
test_data = [mol_desc_test, morgan_fp_test, maccs_keys_test, rdkit_fp_test]
best_params = [best_params_moldesc, best_params_morganfp, best_params_maccs, best_params_rdkitfp]
trained_model_directory = './Models/rf model/Trained rf_models'
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

# mol_desc rf model results
# Avg MSE: 0.851104526842804 ± 0.015602820593581031
# Avg RMSE: 0.9225145345598428 ± 0.008453423486431548
# Avg r2 score: 0.7798814266829304 ± 0.004035310000666449
# Avg MAE: 0.6290012353532142 ± 0.004327767001565231

# morgan_fp rf model results
# Avg MSE: 1.17491922701406 ± 0.040076085509804335
# Avg RMSE: 1.0837820300729633 ± 0.01832316307263766
# Avg r2 score: 0.6636299471723446 ± 0.011473465315839313
# Avg MAE: 0.7512752220097996 ± 0.013319236110026736

# maccs_keys rf model results
# Avg MSE: 1.0983146743475831 ± 0.0308706371107939
# Avg RMSE: 1.047903160317132 ± 0.014616461437361875
# Avg r2 score: 0.6855612228164987 ± 0.008838018475679343
# Avg MAE: 0.7353443296079566 ± 0.006422402560825984

# rdkit_fp rf model results
# Avg MSE: 1.548330261161582 ± 0.023598170511467877
# Avg RMSE: 1.2442831595892432 ± 0.00946994847884592
# Avg r2 score: 0.5567253307572723 ± 0.006755968988397075
# Avg MAE: 0.8530388939309194 ± 0.0048147054649542165
