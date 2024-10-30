import pandas as pd
import numpy as np
import pickle
import utils.config as config
import shap

headers = config.headers
    
def run_model(X) -> str:
    
    with open(config.model_path, 'rb') as model:
        clf = pickle.load(model) 
        
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X.values.reshape(1, -1))
    
    ft_imp_dict = dict(zip(headers, list(shap_values[0])))
    
    ft_imp_dict_sorted = dict(sorted(ft_imp_dict.items(), key=lambda item: item[1], reverse=True))
    
    ft_imp_npiq = [ft for ft in ft_imp_dict_sorted.keys() if 'NPIQ' in ft][0]
    
    npi_priority =  ft_imp_npiq.split('SEV_')[0].split('_')[0]
    
    
    return config.npi_class[npi_priority]


if __name__ == '__main__':
    main_df = pd.read_csv('./data/test_predictions_20241028.csv')
    main_df['npi_priority'] = main_df.apply(lambda x: run_model(x), axis=1)
    