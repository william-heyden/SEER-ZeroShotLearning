# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import scipy
#from torchvision import datasets
#import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from scipy import linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import time
import math
import warnings
import random
from time import gmtime, strftime

import model_class as mc
from load_data import awa2, apy, cub
import engine as eg
import hyper_file
#import hyperbolic_NN as hh

# + jupyter={"outputs_hidden": true}
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epoche", type = int, default=90, help="Number of epoches")
parser.add_argument("-b", "--batch", type = int, default=128, help="Batch size")
parser.add_argument("-z", "--zdim", type = int, default=48, help="Dimension of latent semantic")
parser.add_argument("-c", "--comb", type = int, default=1, help="Combinational training rate")
parser.add_argument("-g", "--generalized", action='store_true', help="generalized setting")
parser.add_argument("-r", "--random_split", action='store_true', help="random unseen split")
parser.add_argument("-d", "--dataset", type=str, default="awa", help="dataset to use (cub, awa, sun)")
parser.add_argument("-k", "--comment", type=str, default=strftime("%b%d_%H%M", gmtime()), help="comment to distinquish runs")
parser.add_argument("-s", "--source", type=str, default="OG", help="semantic embedding source")
parser.add_argument("-l", "--learning_rate", type=int, default=1, help="learning rate coeficient")
#args, unknown = parser.parse_known_args()
args = vars(parser.parse_args())

# +
try:
    b_size = args['batch']
except:
    NameError
    b_size = 128

try:
    n_epochs = args['epoche']
except:
    NameError
    n_epochs = 90

try:
    z_dims = args['zdim']
except:
    NameError
    z_dims = 48
    
try:
    comb = args['comb']
except:
    NameError
    comb = 1
    
try:
    gen = args['generalized']
except:
    NameError
    gen = True
    
try:
    rand = args['random_split']
except:
    NameError
    rand = False

try:
    df = args['dataset']
except:
    NameError
    df = 'awa'
    
try:
    com = args['comment']
except:
    NameError
    com = strftime("%b%d_%H%M", gmtime())
    
try:
    df_source = args['source']
except:
    NameError
    df_source = 'OG'
    
try:
    lr_coef = args['learning_rate']
except:
    NameError
    lr_coef = 1
# -

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
print(f"Using device: {device}")

# ## Load data

GENERALIZED = gen
random_split = rand
loader, s_true, cls_true, semantic, lbls, x_cls = eg.load_data_split(df, 
                                                                     GENERALIZED,
                                                                     random_split,
                                                                     b_size,
                                                                     df_source)
train_loader, test_loader = loader[0], loader[1]

# +
#calculate semantic distance for random split
# Inside unseen classes
#intra_classes_distance = pdist(s_true[1].cpu())
#print(f'intra class (unseen) dist mean : {intra_classes_distance.mean():.5f}')
#print(f'                           std : {intra_classes_distance.std():.5f}')
# Between seen and unseen classes
#class_vice_distance = []
#for i,s in enumerate(s_true[1].cpu()):
#    temp_arr = np.concatenate([s_true[0].cpu(), s.unsqueeze(dim=0)])
#    temp_dist = pdist(temp_arr)
#    class_vice_distance.append(temp_dist[-1])
#    
#print(f'seen vs unseen dist   average : {np.average(class_vice_distance):.5f}')
#print(f'                          std : {np.std(class_vice_distance):.5f}')
#print(f'                          min : {np.min(class_vice_distance):.5f}')
#print(f'                          max : {np.max(class_vice_distance):.5f}')
# -

hyper_tune = hyper_file.get_params(df, z_dims, n_epochs, comb, df_source, com)


# +
#print(hyper_tune)
# -

def train_models(z_d, n_ep, comb_training, patience, early_delta,
                 lr_vae, lr_wgan, lr_cvae, lr_comb, calibrated_stacking,
                 _normal_init, _save_metric, _validation, _print_acc_viz, _allways_save_best, _record_best,
                 vae_prior_w,cvae_prior_w, src, com,
                 device=device):
    z_dim = z_d
    conditional_dim = semantic[0].shape[1]
    save_res = False
    cs = calibrated_stacking
    
    # Correlation between accuracy and validation loss
    corr_val_vae, corr_val_wgan_g, corr_val_wgan_d, corr_val_cvae = [],[],[],[]
    corr_acc = []
    
    #lr_vae = {'lr': 0.010, 'g':0.90}
    #lr_wgan = {'lr_g':0.100, 'lr_d':0.010, 'g_g':0.5, 'g_d':0.1, 'lb':5}
    #lr_cvae = {'lr':0.010, 'g':0.90}
    #lr_comb = {'lr':0.010, 'g':1/3}
    if _print_acc_viz:
        print('_'*65)
        print(f'Initiating models with z-dimension = {z_dim}.', flush=True)
        print(f'batch size: {b_size} - generalize: {GENERALIZED} - random split: {random_split} - dataset: {df} - source: {src}')
        print(f'training on {device}')
        print(f'Comment: {com}')

    vae = mc.vae_simple(hidden_layers=[64,54,48], input_dim=conditional_dim, latent_dim=z_dim, p=lr_vae['drop'])
    generator = mc.generator_simple_dense([256, 512, 2048], z_dim, 2048, lr_wgan['drop'])
    discriminator = mc.discriminator_simple_dense([512, 256, 128], 2048, lr_wgan['drop'])
    cvae = mc.cvae_simple(hidden_layers=[512,256,128], input_dim=2048, latent_dim=conditional_dim,
                          conditional_dim=conditional_dim, p=lr_cvae['drop'])

    vae.apply(lambda m: eg.weights_init(m, _normal_init))
    generator.apply(lambda m: eg.weights_init(m, _normal_init))
    discriminator.apply(lambda m: eg.weights_init(m, _normal_init))
    cvae.apply(lambda m: eg.weights_init(m, _normal_init))

    #loss_vae = nn.CosineSimilarity()
    #loss_vae = nn.BCELoss()
    loss_vae = nn.MSELoss(reduction='sum')
    #loss_vae = nn.NLLLoss()
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=lr_vae['lr']/lr_coef, weight_decay=lr_vae['weight'])
    scheduler_vae = torch.optim.lr_scheduler.ExponentialLR(optimizer_vae, gamma=lr_vae['g'])

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr_wgan['lr_g']/lr_coef, weight_decay=lr_wgan['weight'])
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr_wgan['lr_d']/lr_coef, weight_decay=lr_wgan['weight'])
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=lr_wgan['g_g'])
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=lr_wgan['g_d'])

    #loss_cvae = nn.BCELoss()
    loss_cvae = nn.MSELoss(reduction='sum')
    optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=lr_cvae['lr']/lr_coef, weight_decay=lr_cvae['weight'])
    scheduler_cvae = torch.optim.lr_scheduler.ExponentialLR(optimizer_cvae, gamma=lr_cvae['g'])

    #optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr = 0.1)
    optimizer_combined = torch.optim.Adam(
        list(vae.encoder.parameters()) + list(generator.parameters()) + list(cvae.encoder.parameters()),
        lr=lr_comb['lr']/lr_coef,
        weight_decay=lr_comb['weight'])
    scheduler_combined = torch.optim.lr_scheduler.ConstantLR(optimizer_combined, factor=lr_comb['g'], total_iters=10)
    
    #Classification loss (in VAE, WGAN, or CVAE??)
    xtr_cls, xte_cls, ytr_cls, yte_cls = train_test_split(x_cls, lbls[0].cpu(), test_size=.2)
    gen_cls = eg.train_cls(xtr_cls, ytr_cls)
    
    if _print_acc_viz:
        print(f'classifier accuracy (seen): {gen_cls.score(xte_cls,yte_cls)*100:.3f}%')
    
    early_stopper = eg.EarlyStopper(patience=patience, min_delta=early_delta)
    
    vae_metric = []
    wgan_metric = []
    cvae_metric = []
    combined_metric = []
    
    best_c_s, best_c_u, best_g_s, best_g_u = 0,0,0,0
    
    if _print_acc_viz:
        print(f'training with {n_ep} epochs', flush=True)
    for e in range(n_ep):
            
        # TRAINING
        # _Fisher
        #vae_metric.append(
        #    eg.train_vae_fisher(vae, train_loader, loss_vae, optimizer_vae,
        #                        scheduler_vae, eg.fisher_criterion, verbose=False)
        #)
        vae_metric.append(
            eg.train_vae(vae, train_loader, loss_vae, optimizer_vae,
                                scheduler_vae, verbose=False)
        )
        
        # _WGAN_CLS
        wgan_metric.append(
            eg.train_wgan_cls(generator, discriminator,gen_cls, train_loader, optimizer_G, optimizer_D, 
                      scheduler_G, scheduler_D, lb_term=lr_wgan['lb'], z_dim=z_dim, verbose=False)
        )
        
        cvae_metric.append(
            eg.train_cvae(cvae, train_loader, optimizer_cvae, loss_cvae, scheduler_cvae, 
                          cvae_prior_w, verbose=False)
        )
        
        for i in range(comb_training):
            #combined_metric.append(
            #    eg.train_combined(vae.encoder, generator, cvae.encoder,
            #                      train_loader, optimizer_combined, scheduler_combined)
            #)
            combined_metric.append(
                eg.train_combined_cond(vae.encoder, generator, cvae.encoder,
                                  train_loader, optimizer_combined, scheduler_combined, loss_fn = nn.MSELoss())
            )
        
        acc_seen = eg.unseen_accuracy(vae, generator, cvae,
                                      semantic[0], lbls[0], s_true[0],
                                      cls_true[0])

        acc_unseen = eg.unseen_accuracy(vae, generator, cvae,
                                        semantic[1], lbls[1], s_true[1],
                                        cls_true[1])
        if GENERALIZED:
            acc_gen, acc_gen_avg = eg.generalized_accuracy(vae, generator, cvae,
                                                           semantic[2], lbls[2], s_true[2],
                                                           cls_true[2], cls_true[0], cs)
         
        best_acc = 0.0
        if _allways_save_best and (acc_gen_avg>best_acc):
            torch.save({
                'embeding_model':vae.state_dict(),
                'generator_model': generator.state_dict(),
                'discriminator_model': discriminator.state_dict(),
                'classifying_model': cvae.state_dict()
            }, './models_{}_{}_{}'.format(df, src, com))
        
        if _print_acc_viz:
            if (acc_unseen+acc_seen) != 0.0:
                harm = (2*acc_seen*acc_unseen)/(acc_unseen+acc_seen)
            else:
                harm = 0.0
            print(f'e: {e:2d} - harmonic: {harm*100:.3f}%', flush=True)
            print('\t_conventional_')
            print(f'\t   acc seen {int(acc_seen*100):2d}% |' +  '#'*int(acc_seen*100), flush=True)
            print(f'\t acc unseen {int(acc_unseen*100):2d}% |' +  '#'*int(acc_unseen*100), flush=True)
            if GENERALIZED:
                print('\t_generalized_')
                print(f'\t   acc seen {int(acc_gen*100):2d}% |' +  '#'*int(acc_gen*100), flush=True)
                print(f'\t acc unseen {int(acc_gen_avg*100):2d}% |' +  '#'*int(acc_gen_avg*100), flush=True)
                    

        #VALIDATION
        if _validation:
            vae_val = eg.val_vae(vae, test_loader, loss_vae, verbose=False)
            wgan_val_G, wgan_val_D = eg.val_wgan(generator, discriminator, test_loader, verbose=False)
            cvae_val = eg.val_cvae(cvae, test_loader, loss_cvae, verbose=False)
            val_loss = vae_val + wgan_val_D + wgan_val_G + cvae_val
            #print('-'*30)
            #print(f'total val loss: {val_loss:.5f}')
            
            corr_val_vae.append(vae_val)
            corr_val_wgan_g.append(wgan_val_G)
            corr_val_wgan_d.append(wgan_val_D)
            corr_val_cvae.append(cvae_val)
            corr_acc.append(acc_gen_avg)
            
        if _record_best:
            if acc_seen > best_c_s:
                best_c_s = acc_seen
                e_c_s = e
            if acc_unseen > best_c_u:
                best_c_u = acc_unseen
                e_c_u = e
            if acc_gen > best_g_s:
                best_g_s = acc_gen
                e_g_s = e
            if acc_gen_avg > best_g_u:
                best_g_u = acc_gen_avg
                e_g_u = e
            
        


    print('__ Training completed __')
    
    if _record_best:
        print(f'  Best conventional seen: {100*best_c_s:.3f}% [epoch: {e_c_s}]')
        print(f'Best conventional unseen: {100*best_c_u:.3f}% [epoch: {e_c_u}]')
        print(f'   Best generalized seen: {100*best_g_s:.3f}% [epoch: {e_g_s}]')
        print(f' Best generalized unseen: {100*best_g_u:.3f}% [epoch: {e_g_u}]')
    
    if _validation:
        print('... saving validation', flush=True)
        eg.save_array(corr_val_vae, 'corr_val_vae_{}'.format(com))
        eg.save_array(corr_val_wgan_g, 'corr_val_wgan_g_{}'.format(com))
        eg.save_array(corr_val_wgan_d, 'corr_val_wgan_d_{}'.format(com))
        eg.save_array(corr_val_cvae, 'corr_val_cvae_{}'.format(com))
        eg.save_array(corr_acc, 'corr_acc_{}'.format(com))
    
    if _save_metric:
        print('... saving results', flush=True)
        eg.save_array(vae_metric, 'VAE_loss_b{}_g{}_r{}'.format(b_size, GENERALIZED,random_split))
        eg.save_array(wgan_metric, 'WGAN_loss_b{}_g{}_r{}'.format(b_size,GENERALIZED,random_split))
        eg.save_array(cvae_metric, 'CVAE_loss_b{}_g{}_r{}'.format(b_size,GENERALIZED,random_split))
        eg.save_array(combined_metric, 'COMBINED_loss_b{}_g{}_r{}'.format(b_size,GENERALIZED,random_split))
    return vae, generator, discriminator, cvae, save_res

# + jupyter={"outputs_hidden": true}
warnings.filterwarnings('ignore') 
m1, m2, m3, m4, _save_score = train_models(**hyper_tune)


# +
#emb_model = m1
#gen_model = m2
#disc_model = m3
#class_model = m4
# -

def score_models(emb_model, gen_model, disc_model, class_model, _save_models=True):
    emb_score = {'rank':[], 'ne':[], 'psnr':[]}
    gan_score = {'d_real_acc':[], 'd_gen_acc':[], 'geo_score':[], 'fid':[]}
    class_score = {'dist':[]}
    accuracy_score = {'overall_acc':[], 'precision': [], 'recall': []}

    emb_model.to(device).eval()
    gen_model.to(device).eval()
    disc_model.to(device).eval()
    class_model.to(device).eval()
    
    test_input = torch.tensor(Ste, dtype=torch.float, device=device)
    # No need to loop batch?
    with torch.inference_mode():
        emb_space,_,_ = emb_model.encoder(test_input)
        gen_space = gen_model(emb_space)
        class_space,_,_ = class_model.encoder(gen_space, test_input)
    
    truth = torch.unique(test_input, dim=0)
    
            
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred_lst = []
    for i in range(class_space.shape[0]):
        dist = 1-cos(class_space[i].unsqueeze(dim=0), truth)
        # true class id??
        pred_lst.append(class_lbls_test[np.argmin(dist.detach().cpu().numpy())])
    
    accuracy_score['precision'] = precision_score(y_test, pred_lst, average='weighted', zero_division=0)
    accuracy_score['recall'] = recall_score(y_test, pred_lst, average='weighted', zero_division=0)
    acc = 0
    for i in range(len(pred_lst)):
        if pred_lst[i] == y_test[i]:
            acc += 1
    accuracy_score['overall_acc'] = acc/len(pred_lst)
    #confuse = confusion_matrix(y_test, pred_lst)
    
    emb_score['rank'] = eg.RankMe(emb_space.cpu()).item()
    emb_score['ne'] = eg.NEsum(emb_space.cpu()).item()
    emb_score['psnr'] = eg.PSNR(emb_model(test_input)[0].detach().cpu(), test_input.cpu()).item()
    
    x_real = torch.tensor(Xte, dtype=torch.float, device=device)
    x_gen = gen_space.clone().detach()
    label_ones = torch.ones(x_real.size(0), 1, device=device)
    label_zeros = torch.zeros(x_real.size(0), 1, device=device)
    with torch.inference_mode():
        pred_gen = disc_model(x_gen)
        pred_real = disc_model(x_real)
    gan_score['d_real_acc'] = torch.sum(torch.eq(label_ones, torch.round(pred_real.squeeze())))/x_real.size(0)
    gan_score['d_gen_acc'] = torch.sum(torch.eq(label_zeros, torch.round(pred_gen.squeeze())))/x_gen.size(0)
    gan_score['geo_score'] = eg.Geometry_Score(np.array(gen_space.detach().cpu()), Xte).item()
    gan_score['fid'] = eg.calculate_fid(gen_space, torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().item()
    
    dist_score = 0.0
    for l in np.unique(y_test):
        class_temp = class_space[np.where(y_test==l)]
        proto_temp = test_input[np.where(y_test==l)][0]
        dist_score += eg.prototype_distance(class_temp.clone().detach(), proto_temp.clone().detach()).clone().detach().cpu()
    class_score['dist'] = dist_score/len(np.unique(y_test))
    
    if _save_models:
        torch.save({
            'embeding_model':emb_model.state_dict(),
            'generator_model': gen_model.state_dict(),
            'discriminator_model': disc_model.state_dict(),
            'classifying_model': class_model.state_dict()
        }, '/models.pt')
        
        
    
    return emb_score, gan_score, class_score, accuracy_score, pred_lst, y_test

if _save_score:
    
    embedding_score, generative_score, classification_score, acc_score, prediction_lst, true_lst = score_models(m1,m2,m3,m4)
    
    eg.save_array(embedding_score, 'embedding_score_CUB_Z{}_B{}'.format(z_dims,b_size))
    eg.save_array(generative_score, 'generative_score_CUB_Z{}_B{}'.format(z_dims,b_size))
    eg.save_array(classification_score, 'classification_CUB_score_Z{}_B{}'.format(z_dims,b_size))
    eg.save_array(acc_score, 'accuracy_score_CUB_Z{}_B{}'.format(z_dims,b_size))

    eg.save_array(prediction_lst, 'prediction_list_CUB_Z{}_B{}'.format(z_dims,b_size))
    eg.save_array(true_lst, 'true_list_CUB_Z{}_B{}'.format(z_dims,b_size))
