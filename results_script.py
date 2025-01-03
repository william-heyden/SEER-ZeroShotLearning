import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

import model_class as mc
import engine as eg
import hyper_train
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

device = eg.get_device()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', "--dataset", default='awa')
parser.add_argument('-b', "--batch_size", type=int, default=128)
parser.add_argument('-s', "--source", default='OG') #vgse # w2v # OG
args = parser.parse_args()

df = args.dataset
b_size = args.batch_size
num_epochs = 20
df_source = args.source

loader, s_true, cls_true, semantic, lbls, x_cls = eg.load_data_split(df,True,False,b_size,df_source)
train_loader, test_loader = loader[0], loader[1]

conditional_dim = semantic[0].shape[1]
visual_dim = x_cls.shape[1]
z_dim = 48
calibrated_stacking = 0
_normal_init = True
_allways_save_best = False
_record_best = True
_print_loop = True
_flush= 1
_fileId = datetime.now().strftime("%m_%d_%H%M")
_save_metric = False

# Inference
shuffle = True
noise = False

lrs, betas, comb_training = hyper_train.get_params(df)
lr_vae, lr_wgan, lr_cvae, lr_comb = lrs[0], lrs[1], lrs[2], lrs[3] 
vae_beta, cvae_beta = betas[0], betas[1] 

#print(lrs, betas, comb_training)
print('-'*55, flush=True)
print(f' RUN ID: {_fileId} | DF: {df} | source: {df_source} | batch: {b_size} | Shuffel: {shuffle} | Noise: {noise} |', flush=True)
print(f'Visaul space accuracy - _ALTERNATIVE_ Generalized dataset', flush=True)
print('SEER true space for classification = SS (true generalized) [line: 2078/2046]', flush=True)
print('_'*55, flush=True)
vae = mc.vae_simple(hidden_layers=[64,54,48], input_dim=conditional_dim, latent_dim=z_dim, p=lr_vae['drop'])
generator = mc.generator_simple_dense([256, 512, 2048], z_dim, visual_dim, lr_wgan['drop'])
discriminator = mc.discriminator_simple_dense_cls([512, 256, 128], visual_dim, lr_wgan['drop'], len(cls_true[0]))
cvae = mc.cvae_simple(hidden_layers=[512,256,128], input_dim=visual_dim, latent_dim=conditional_dim,
                        conditional_dim=conditional_dim, p=lr_cvae['drop'])

vae.apply(lambda m: eg.weights_init(m, _normal_init))
generator.apply(lambda m: eg.weights_init(m, _normal_init))
discriminator.apply(lambda m: eg.weights_init(m, _normal_init))
cvae.apply(lambda m: eg.weights_init(m, _normal_init))

optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae['lr'], weight_decay=lr_vae['weight'])
scheduler_vae = torch.optim.lr_scheduler.ExponentialLR(optimizer_vae, gamma=lr_vae['g'])

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr_wgan['lr_g'], weight_decay=lr_wgan['weight'])
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr_wgan['lr_d'], weight_decay=lr_wgan['weight'])
scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=lr_wgan['g_g'])
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=lr_wgan['g_d'])

#loss_cvae = nn.BCELoss()
loss_cvae = nn.MSELoss(reduction='sum')
optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=lr_cvae['lr'], weight_decay=lr_cvae['weight'])
scheduler_cvae = torch.optim.lr_scheduler.ExponentialLR(optimizer_cvae, gamma=lr_cvae['g'])

#optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr = 0.1)
optimizer_combined = torch.optim.Adam(
    list(vae.encoder.parameters()) + list(generator.parameters()) + list(cvae.encoder.parameters()),
    lr=lr_comb['lr'],
    weight_decay=lr_comb['weight'])
scheduler_combined = torch.optim.lr_scheduler.ConstantLR(optimizer_combined, factor=lr_comb['g'], total_iters=10)

#xtr_cls, xte_cls, ytr_cls, yte_cls = train_test_split(x_cls, lbls[0].cpu(), test_size=.5)
#gen_cls = eg.train_cls(xtr_cls, ytr_cls, df, use_saved=True)
unique_label = sorted(set(cls_true[0]))
label_to_index = {label.item(): index for index, label in enumerate(unique_label)}
classification_loss = nn.CrossEntropyLoss()

# Get generalized data data (unseen + .3 x seen)
x_seen, x_unseen, s_seen, s_unseen, y_seen, y_unseen = eg.get_X(train_loader, test_loader, lbls[2]) # 2 -> generalized, 0-> unseen, 1-> seen
S_true, X_true, Y_true = torch.cat([s_seen, s_unseen], dim=0), torch.cat([x_seen, x_unseen], dim=0), torch.cat([y_seen, y_unseen], dim=0)

vae_metric = []
wgan_metric = []
cvae_metric = []
combined_metric = []
analysis_vae = {"train_bce": [], "train_kld": [], 'train_clp':[], 'train_acc':[],
                "test_bce": [], "test_kld": [],  'test_clp':[], 'test_acc':[]}
analysis_cvae = {"train_bce": [], "train_kld": [], 'train_clp':[], 'train_acc':[],
                "test_bce": [], "test_kld": [],  'test_clp':[], 'test_acc':[]}
def train_step():
    best_acc = 0.0
    best_c_s, best_c_u, best_g_s, best_g_u = 0,0,0,0
    for e in range(num_epochs):
        
        # VAE
        vae_res = eg.train_vae(vae, train_loader, optimizer_vae,
                         scheduler_vae, prior_weight=vae_beta)
        vae_metric.append(vae_res[0])
        analysis_vae['train_bce'].append(vae_res[1][0])
        analysis_vae['train_kld'].append(vae_res[1][1])
        analysis_vae['train_clp'].append(vae_res[1][2])
        
        # WGAN
        wgan_metric.append(
            eg.train_wgan_cls(generator, discriminator, train_loader, optimizer_G, optimizer_D, 
                      None, None, classification_loss, label_to_index, unique_label,
                      lb_term=lr_wgan['lb'], z_encoder=vae, z_dim=z_dim,
                       gen_training_step=lr_wgan['gen_step'], cls_weight=lr_wgan['cls'])
        )
        
        # CVAE
        cvae_res = eg.train_cvae(cvae, train_loader, optimizer_cvae, scheduler_cvae, 
                          prior_weight=cvae_beta, vae=vae, gen=generator)
        cvae_metric.append(cvae_res[0])
        analysis_cvae['train_bce'].append(cvae_res[1][0])
        analysis_cvae['train_kld'].append(cvae_res[1][1])
        analysis_cvae['train_clp'].append(cvae_res[1][2])
        
        # Combined
        for i in range(comb_training):
            combined_metric.append(
                eg.train_combined(vae.encoder, generator, cvae.encoder,
                                  train_loader, optimizer_combined, scheduler_combined)
            )

        # Classification real-generalized visual space
        acc_seen, acc_unseen = eg.generalized_acc_seen_unseen(vae, generator, cvae,
                                       s_true[2],Y_true,X_true, class_labels=None,
                                       shuffle=shuffle, noise=noise, batch_size=b_size,
                                       seen_lables=cls_true[0])
        

        if _allways_save_best and (harm>best_acc):
            torch.save({
                'embeding_model':vae.state_dict(),
                'generator_model': generator.state_dict(),
                'discriminator_model': discriminator.state_dict(),
                'classifying_model': cvae.state_dict()
            }, './models_{}_{}'.format(df,_fileId))
            best_acc = harm
            print('model saved as: _{}'.format(_fileId), flush=True)

        if e%_flush==0 and _print_loop:
            if (acc_unseen+acc_seen) != 0.0:
                harm = (2*acc_seen*acc_unseen)/(acc_unseen+acc_seen)
            else:
                harm = 0.0
            #if (acc_unseen1+acc_seen1) != 0.0:
            #    harm1 = (2*acc_seen1*acc_unseen1)/(acc_unseen1+acc_seen1)
            #else:
            #    harm1 = 0.0

            #print(f'e: {e:3d} - harmonic: {harm*100:.3f}%', flush=True)
            #print('\t_conventional_')
            #print(f'\t   acc seen {int(acc_seen*100):2d}% |' +  '#'*int(acc_seen*100), flush=True)
            #print(f'\t acc unseen {int(acc_unseen*100):2d}% |' +  '#'*int(acc_unseen*100), flush=True)
            #print('\t_generalized_')
            #print(f'\t   acc seen {int(acc_gen*100):2d}% |', flush=True)# +  '#'*int(acc_gen*100)
            #print(f'\t acc unseen {int(acc_gen_avg*100):2d}% |', flush=True)# +  '#'*int(acc_gen_avg*100), flush=True)
            #print('\t_visual space_')
            #print(f'\t   acc seen {int(acc_visual_seen*100):2d}% |', flush=True)# +  '#'*int(acc_visual_seen*100), flush=True)
            #print(f'\t acc unseen {int(acc_visual_unseen*100):2d}% |', flush=True)# +  '#'*int(acc_visual_unseen*100), flush=True)
            #print(f'\t\t Harmonic: {(2*acc_visual_seen*acc_visual_unseen)/(acc_visual_seen+acc_visual_unseen)*100:.3f}%', flush=True)
            #harm_vis = (2*acc_visual_seen*acc_visual_unseen)/(acc_visual_seen+acc_visual_unseen)
            print(f'E:{e:3d} | S: {100*acc_seen:.2f}% U: {100*acc_unseen:.2f}% H: {harm*100:.2f}%',flush=True)# |Loop - S: {int(acc_seen1*100):.2f}% U: {int(acc_unseen1*100):.2f}% H: {harm1*100:.2f}%', flush=True)

        if _record_best:
            if acc_seen > best_c_s:
                best_c_s = acc_seen
                e_c_s = e
            if acc_unseen > best_c_u:
                best_c_u = acc_unseen
                e_c_u = e
            #if acc_seen1 > best_g_s:
            #    best_g_s = acc_seen1
            #    e_g_s = e
            #if acc_unseen1 > best_g_u:
            #    best_g_u = acc_unseen1
            #    e_g_u = e
        


            
    print('__ Training completed __', flush=True)
    
    if _record_best:
        #print(f'  Best conventional seen: {100*best_c_s:.3f}% [epoch: {e_c_s}]', flush=True)
        #print(f'Best conventional unseen: {100*best_c_u:.3f}% [epoch: {e_c_u}]', flush=True)
        #print(f'   Best generalized seen: {100*best_g_s:.3f}% [epoch: {e_g_s}]', flush=True)
        #print(f' Best generalized unseen: {100*best_g_u:.3f}% [epoch: {e_g_u}]', flush=True)
        print(f'M: Best seen: {100*best_c_s:.3f}% [epoch: {e_c_s}]', flush=True)
        print(f'M: Best unseen: {100*best_c_u:.3f}% [epoch: {e_c_u}]', flush=True)
        #print(f'L: Best seen: {100*best_g_s:.3f}% [epoch: {e_g_s}]', flush=True)
        #print(f'L: Best unseen: {100*best_g_u:.3f}% [epoch: {e_g_u}]', flush=True)

    if _save_metric:
        print('... saving results', flush=True)
        eg.save_array(vae_metric, 'VAE_loss_id{}'.format(_fileId))
        eg.save_array(wgan_metric, 'WGAN_loss_id{}'.format(_fileId))
        eg.save_array(cvae_metric, 'CVAE_loss_id{}'.format(_fileId))
        eg.save_array(combined_metric, 'COMBINED_loss_id{}'.format(_fileId))
    
    #eg.plot_metrics(analysis_vae, 'vae', df, kl_weight=vae_beta, latent_dim=z_dim, file_id=_fileId)
    #eg.plot_metrics(analysis_cvae, 'cvae', df, kl_weight=cvae_beta, latent_dim=conditional_dim, file_id=_fileId)
    #eg.plot_lines_from_lists(wgan_metric, df, file_id=_fileId)

train_step()



