def get_params(df, z_dims, n_epochs, comb, df_source, com):
    if df == 'awa':
        hyper_tune = {'z_d': z_dims,
                  'n_ep': n_epochs,
                  'comb_training': comb,
                  'src': df_source,
                  'com': com,
                  'patience': 3,
                  'early_delta': .5,
                  'vae_prior_w' : 0.1,
                  'cvae_prior_w': 0.01,
                  'calibrated_stacking':0,
                  'lr_vae': {'lr': 0.010, 'g':0.90, 'weight':1e-03, 'drop':0.1},
                  #'lr_wgan': {'lr_g':0.100/10, 'lr_d':0.010, 'g_g':0.5, 'g_d':0.1, 'lb':5, 'weight':1e-03, 'drop':0.1},
                  'lr_wgan': {'lr_g':0.100/1000, 'lr_d':0.010/100, 'g_g':0.005, 'g_d':0.005, 'lb':0.2, 'weight':9e-02, 'drop':0.6},
                  'lr_cvae': {'lr':0.010, 'g':0.90, 'weight':9e-03, 'drop':0.1},
                  'lr_comb': {'lr':0.010, 'g':1/3, 'weight':2e-03},
                  '_normal_init': False,
                  '_save_metric': False,
                  '_validation':False,
                  '_print_acc_viz': True,
                  '_allways_save_best': True,
                  '_record_best': True
                 }
    elif df == 'cub':
        hyper_tune = {'z_d': z_dims,
                  'n_ep': n_epochs,
                  'comb_training': comb,
                  'src': df_source,
                  'com': com,
                  'patience': 3,
                  'early_delta': .5,
                  'vae_prior_w' : 0.1,
                  'cvae_prior_w': 0.01,
                  'calibrated_stacking':0,
                  'lr_vae': {'lr': 0.10, 'g':0.40, 'weight':1e-03, 'drop':0.0},
                  'lr_wgan': {'lr_g':0.10, 'lr_d':0.01, 'g_g':0.5, 'g_d':0.1, 'lb':1, 'weight':1e-03, 'drop':0.3},
                  'lr_cvae': {'lr':0.10, 'g':0.50, 'weight':9e-02, 'drop':0.5},
                  'lr_comb': {'lr':0.010, 'g':1/3, 'weight':1e-04},
                  '_normal_init': False,
                  '_save_metric': False,
                  '_validation':False,
                  '_print_acc_viz': False,
                  '_allways_save_best': True,
                  '_record_best': True
                 }
    elif df == 'sun':
        hyper_tune = {'z_d': z_dims,
                  'n_ep': n_epochs,
                  'comb_training': comb,
                  'src': df_source,
                  'com': com,
                  'patience': 3,
                  'early_delta': .5,
                  'vae_prior_w' : 0.1,
                  'cvae_prior_w': 0.01,
                  'calibrated_stacking':0,
                  'lr_vae': {'lr': 0.10, 'g':0.40, 'weight':1e-03, 'drop':0.0},
                  'lr_wgan': {'lr_g':0.10, 'lr_d':0.01, 'g_g':0.5, 'g_d':0.1, 'lb':1, 'weight':1e-03, 'drop':0.3},
                  'lr_cvae': {'lr':0.10, 'g':0.50, 'weight':9e-02, 'drop':0.5},
                  'lr_comb': {'lr':0.010, 'g':1/3, 'weight':1e-04},
                  '_normal_init': False,
                  '_save_metric': False,
                  '_validation':False,
                  '_print_acc_viz': True,
                  '_allways_save_best': True,
                  '_record_best': True
                 }
    else:
        print('no valid dataset')
    
    return hyper_tune