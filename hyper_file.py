
def get_params(df):
    if df.lower() == 'awa':
        lr_vae = {'lr': 0.09, 'g':0.90, 'drop': 0.0, 'weight':5e-02} #done
        lr_wgan = {'lr_g':0.00050, 'lr_d':0.00090, 'g_g':0.2, 'g_d':0.2, 'lb':10, 'drop': 0.0, 'weight':0.001, 'cls':10.0, 'gen_step':1} #done
        lr_cvae = {'lr':0.09, 'g':0.90, 'drop': 0.0, 'weight':0.00010} #done
        lr_comb = {'lr':0.010, 'g':1/3, 'drop': 0.1, 'weight':2e-03}
        beta_vae, beta_cvae, comb_training = 0.01, 0.0010, 1
    elif df.lower() == 'cub':
        lr_vae = {'lr': 0.0010, 'g':0.90, 'drop': 0.0, 'weight':0.000050} #done
        lr_wgan = {'lr_g':0.000050, 'lr_d':0.000050, 'g_g':0.5, 'g_d':0.1, 'lb':10, 'drop': 0.1, 'weight':0.00010, 'cls':10.0, 'gen_step':1} #done
        lr_cvae = {'lr':0.0010, 'g':0.90, 'drop': 0.0, 'weight':0.00010} #done
        lr_comb = {'lr':0.010, 'g':1/3, 'drop': 0.1, 'weight':2e-03}
        beta_vae, beta_cvae, comb_training = 0.001, 0.001, 1
    elif df.lower() == 'sun':
        lr_vae = {'lr': 0.0010, 'g':0.90, 'drop': 0.0, 'weight':0.00010} #done
        lr_wgan = {'lr_g':0.000050, 'lr_d':0.000050, 'g_g':0.5, 'g_d':0.1, 'lb':10, 'drop': 0.1, 'weight':0.000010, 'cls':10.0, 'gen_step':15} #done
        lr_cvae = {'lr':0.0010, 'g':0.5, 'drop': 0.0, 'weight':0.00010} #done
        lr_comb = {'lr':0.010, 'g':1/3, 'drop': 0.1, 'weight':1e-04}
        beta_vae, beta_cvae, comb_training = 0.001, 0.001, 1
    elif df.lower() == 'sun_vit':
        lr_vae = {'lr': 0.010, 'g':0.90, 'drop': 0.1, 'weight':1e-03}
        lr_wgan = {'lr_g':0.10, 'lr_d':0.010, 'g_g':0.5, 'g_d':0.1, 'lb':10, 'drop': 0.3, 'weight':1e-03, 'cls':10.0, 'gen_step':1}
        lr_cvae = {'lr':0.0010, 'g':0.50, 'drop': 0.5, 'weight':9e-02}
        lr_comb = {'lr':0.010, 'g':1/3, 'drop': 0.1, 'weight':1e-04}
        beta_vae, beta_cvae, comb_training = 1.0, 0.02, 1
    elif df.lower() == 'flo':
        lr_vae = {'lr': 0.0005, 'g':0.90, 'drop': 0.0, 'weight':0.050} #done
        lr_wgan = {'lr_g':0.0010, 'lr_d':0.0010, 'g_g':0.5, 'g_d':0.1, 'lb':10, 'drop': 0.2, 'weight':0.0005, 'cls':10.0, 'gen_step':5} #done
        lr_cvae = {'lr':0.0005, 'g':0.50, 'drop': 0.0, 'weight':0.00050}
        lr_comb = {'lr':0.010, 'g':1/3, 'drop': 0.1, 'weight':1e-04}
        beta_vae, beta_cvae, comb_training = 0.0010, 0.010, 10
    elif df.lower() == 'apy':
        lr_vae = {'lr': 0.000010, 'g':0.90, 'drop': 0.0, 'weight':0.0050} #done
        lr_wgan = {'lr_g':0.000050, 'lr_d':0.000050, 'g_g':0.5, 'g_d':0.1, 'lb':10, 'drop': 0.6, 'weight':0.0010, 'cls':10.0, 'gen_step':10} #done
        lr_cvae = {'lr':0.000010, 'g':0.90, 'drop': 0.0, 'weight':0.0010} #done
        lr_comb = {'lr':0.10, 'g':1/3, 'drop': 0.1, 'weight':2e-03}
        beta_vae, beta_cvae, comb_training = 0.0080, 0.001, 1
    elif df.lower() == 'nus':
        lr_vae = {'lr': 0.010, 'g':0.90, 'drop': 0.1, 'weight':1e-03}
        lr_wgan = {'lr_g':0.001, 'lr_d':0.010, 'g_g':0.5, 'g_d':0.1, 'lb':10, 'drop': 0.6, 'weight':9e-02, 'cls':10.0, 'gen_step':1}
        lr_cvae = {'lr':0.010, 'g':0.90, 'drop': 0.1, 'weight':9e-03}
        lr_comb = {'lr':0.010, 'g':1/3, 'drop': 0.1, 'weight':2e-03}
        beta_vae, beta_cvae, comb_training =1.0, 0.02, 1
    else:
        print('No valid dataset selected')
    return [lr_vae, lr_wgan, lr_cvae, lr_comb], [beta_vae, beta_cvae], comb_training
