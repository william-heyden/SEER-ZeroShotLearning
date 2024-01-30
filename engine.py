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

# +
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import gs #https://github.com/KhrulkovV/geometry-score/tree/master
import scipy
#from scipy import linalg@
import model_class as mc
import load_data as ld
import diffusion_model as dm

if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
print(f"Using device: {device}")


# -

def train_vae(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, prior_weight=1.0,
              device: torch.device = device, verbose:bool=True):
    train_loss = 0
    model.to(device).train()
    for batch, (x,s,y) in enumerate(data_loader):
        # Send data to GPU
        s = s.to(device)
        # 1. Forward pass
        s_pred, s_mu, s_var = model(s)
        s_pred = torch.nan_to_num(s_pred)
        # 2. Calculate loss
        loss_rec = torch.mean(loss_fn(s_pred, s))
        #loss_kl = -0.5 * torch.sum(1 + s_var - s_mu.pow(2) - s_var.exp())
        loss_kl = -0.5*torch.mean(1 + s_var - s_mu.pow(2) - s_var.exp())
        loss = loss_rec + prior_weight*loss_kl
        train_loss += float(loss)
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
    # Calculate loss and accuracy per epoch and print out what's happening
    scheduler.step()
    train_loss /= len(data_loader)
    if verbose:
        print(f"Train loss: {train_loss:.5f}")#| Train accuracy: {train_acc:.2f}%")
    return train_loss



def val_vae(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module,
            devide: torch.device = device, verbose:bool=True):
    model.to(device)
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for batch, (_,s,_) in enumerate(data_loader):
            s = s.to(device)
            s_pred, s_mu, s_var = model(s)
            s_pred = torch.nan_to_num(s_pred)
            loss_rec = torch.mean(loss_fn(s_pred, s))
            loss_kl = -0.5 * torch.sum(1 + s_var - s_mu.pow(2) - s_var.exp())
            val_loss += (loss_rec + loss_kl)
    val_temp = val_loss.detach()
    val_temp /= len(data_loader)
    if verbose:
        print(f'Test loss (VAE): \t {val_temp:.5f}')
    return val_temp.item()


def train_wgan_OLD(gen: nn.Module, disc: nn.Module, data_loader: torch.utils.data.DataLoader,
               opti_g:torch.optim.Optimizer, opti_d: torch.optim.Optimizer, scheduler_g: torch.optim.lr_scheduler,
               scheduler_d: torch.optim.lr_scheduler, z_input:torch.tensor=None, z_dim:int = 24, verbose:bool=False,
               device:torch.device=device, gen_training_step:int=5):
    train_loss_g, train_loss_d = 0.0, 0.0
    gen.to(device).train()
    disc.to(device).train()
    for batch, (x,_,_) in enumerate(data_loader):
        real_img = x.to(device)
        if not z_input:
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
        else:
            z = z_input.to(device)
            # z_input = z,_,_ = vae.encoder(s)
        fake_img = gen(z).detach()
        
        #Train Discriminator
        #loss_d = -torch.mean(torch.log(disc(real_img)) + torch.log(1-disc(fake_img)))
        
        loss_d = -torch.mean((disc(real_img)) + (1-disc(fake_img)))
        #loss_d = -torch.mean(disc(real_img)) + torch.mean(disc(fake_img))
        opti_d.zero_grad()
        loss_d.backward()
        opti_d.step()
        train_loss_d += loss_d
        
        #Train Generator
        if batch % gen_training_step == 0:
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
            fake_img = gen(z)
            # Non-Saturating GAN Loss:
            #loss_g = -torch.mean(torch.log(disc(fake_img)))
            loss_g = -torch.mean(disc(fake_img))
            opti_g.zero_grad()
            loss_g.backward()
            opti_g.step()
            train_loss_g += loss_g
    
    scheduler_g.step()
    scheduler_d.step()
    train_loss_d/=len(data_loader)
    train_loss_g/=len(data_loader)
    if verbose:
        print(f'Discriminator loss: {train_loss_d:.5f} | Generator loss: {train_loss_g:.5f}')
    return [train_loss_g.item(), train_loss_d.item()]



def calculate_gradient_penalty(real_images, fake_images, discriminator,lambda_term, b_size, device=device):        
        eta = torch.FloatTensor(real_images.size(0),1).uniform_(0,1)
        eta = eta.expand(real_images.size(0), real_images.size(1))
        eta = eta.to(device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = interpolated.to(device).requires_grad_(True)
        prob_interpolated = discriminator(interpolated).to(device)

        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size(), device=device),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty


def train_wgan(gen: nn.Module, disc: nn.Module, data_loader: torch.utils.data.DataLoader,
               opti_g:torch.optim.Optimizer, opti_d: torch.optim.Optimizer, scheduler_g: torch.optim.lr_scheduler,
               scheduler_d: torch.optim.lr_scheduler, lb_term:float=0.0, z_input:torch.tensor=None, z_dim:int = 24, verbose:bool=False,
               device:torch.device=device, gen_training_step:int=1):
    
    train_loss_g, train_loss_d = 0.0, 0.0
    
    gen.to(device).train()
    disc.to(device).train()
    
    #b_loss_g, b_loss_d = [],[]
    
    #one = torch.tensor(1, dtype=torch.float)
    #mone = one * -1    
        
    for batch, (x,_,_) in enumerate(data_loader):
        real_img = x.to(device)
        if not z_input:
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
        else:
            z = z_input.to(device)
            # z_input = z,_,_ = vae.encoder(s)
        fake_img = gen(z).detach()
        
        #Train Discriminator [1 - real, 0 - fake]
        d_loss_real = torch.mean(disc(real_img))
        d_loss_fake = torch.mean(disc(fake_img))
        
        grad_pen = calculate_gradient_penalty(real_img, fake_img, disc, lb_term, real_img.size(0))
        
        d_loss = d_loss_fake - d_loss_real + grad_pen
        opti_d.zero_grad()
        d_loss.backward()
        opti_d.step()
        
        train_loss_d += d_loss 
        
        #Train Generator
        if batch % gen_training_step == 0:
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
            f_img = gen(z)
            # Non-Saturating GAN Loss:
            #loss_g = -torch.mean(torch.log(disc(fake_img)))
            loss_g = -torch.mean(disc(f_img))
            opti_g.zero_grad()
            loss_g.backward()
            opti_g.step()
            train_loss_g += loss_g
        
        #b_loss_g.append(loss_g)
        #b_loss_d.append(d_loss)
        
    scheduler_g.step()
    scheduler_d.step()
    train_loss_d/=len(data_loader)
    train_loss_g/=len(data_loader)
    if verbose:
        print(f'Discriminator loss: {train_loss_d:.5f} | Generator loss: {train_loss_g:.5f}')
    return [train_loss_g.item(), train_loss_d.item()]


def train_cls(x,y):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(x, y)
    return clf


def train_wgan_cls(gen: nn.Module, disc: nn.Module, cls_model, data_loader: torch.utils.data.DataLoader,
               opti_g:torch.optim.Optimizer, opti_d: torch.optim.Optimizer, scheduler_g: torch.optim.lr_scheduler,
               scheduler_d: torch.optim.lr_scheduler, lb_term:float=0.0, z_input:torch.tensor=None, z_dim:int = 48, verbose:bool=False,
               device:torch.device=device, gen_training_step:int=1):
    
    train_loss_g, train_loss_d = 0.0, 0.0
    
    gen.to(device).train()
    disc.to(device).train()
    #b_loss_g, b_loss_d = [],[]
    
    #one = torch.tensor(1, dtype=torch.float)
    #mone = one * -1    
        
    for batch, (x,_,_) in enumerate(data_loader):
        real_img = x.to(device)
        if not z_input:
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
        else:
            z = z_input.to(device)
            # z_input = z,_,_ = vae.encoder(s)
        fake_img = gen(z).detach()
        
        #Train Discriminator [1 - real, 0 - fake]
        d_loss_real = torch.mean(disc(real_img))
        d_loss_fake = torch.mean(disc(fake_img))
        
        grad_pen = calculate_gradient_penalty(real_img, fake_img, disc, lb_term, real_img.size(0))
        
        d_loss = d_loss_fake - d_loss_real + grad_pen
        opti_d.zero_grad()
        d_loss.backward()
        opti_d.step()
        
        train_loss_d += d_loss 
        
        #Train Generator
        if batch % gen_training_step == 0:
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
            f_img = gen(z)
            #
            f_img = torch.nan_to_num(f_img)
            # Classification loss
            cls_pred = cls_model.predict_log_proba(f_img.cpu().detach().numpy())
            cls_loss = np.mean([cls_pred[i][np.argmax(cls_pred[i])] for i in range(x.shape[0])])
            
            loss_g = -torch.mean(disc(f_img)) + cls_loss
            opti_g.zero_grad()
            loss_g.backward()
            opti_g.step()
            train_loss_g += loss_g
        
        #b_loss_g.append(loss_g)
        #b_loss_d.append(d_loss)
        
    scheduler_g.step()
    scheduler_d.step()
    train_loss_d/=len(data_loader)
    train_loss_g/=len(data_loader)
    if verbose:
        print(f'Discriminator loss: {train_loss_d:.5f} | Generator loss: {train_loss_g:.5f}')
    return [train_loss_g.item(), train_loss_d.item()]


def val_wgan(gen: nn.Module, disc:nn.Module, data_loader: torch.utils.data.DataLoader,
            device: torch.device = device, z_dim:int=48, verbose:bool=True):
    gen.to(device)
    disc.to(device)
    gen.eval()
    disc.eval()
    val_loss_gen, val_loss_disc = 0.0, 0.0
    b_loss_g, b_loss_d = [],[]
    with torch.inference_mode():
        for batch, (x,_,_) in enumerate(data_loader):
            x = x.to(device)
            z = torch.tensor(np.random.normal(0,1, (x.shape[0],z_dim)),dtype=torch.float).to(device)
            gen_fake = gen(z)
            
            val_loss_disc += -torch.mean((disc(x)) + (1-disc(gen_fake)))
            val_loss_gen += -torch.mean(disc(gen_fake))
            
            b_loss_g.append(-torch.mean(disc(gen_fake)).item())
            b_loss_d.append(-torch.mean((disc(x)) + (1-disc(gen_fake))).item())
            
    val_loss_disc, val_loss_gen = val_loss_disc.detach(), val_loss_gen.detach()
    val_loss_disc /= len(data_loader)
    val_loss_gen /= len(data_loader)
    if verbose:
        print(f'Test loss (gen): \t {val_loss_gen:.5f}\nTest loss (disc): \t {val_loss_disc:.5f}')
    #return val_loss_gen.item(), val_loss_disc.item()
    return b_loss_g, b_loss_d


# +
def train_cvae(model: nn.Module, data_loader: torch.utils.data.DataLoader, opti: torch.optim.Optimizer,
               loss_rec: torch.nn.Module, scheduler: torch.optim.lr_scheduler, prior_weight=1.0,
               x_generated: torch.tensor=None, verbose:bool=True, device: torch.device=device):
    train_loss = 0.0
    model.to(device).train()
    for batch, (x,s,_) in enumerate(data_loader):
        s = s.to(device)
        if not x_generated:
            x = x.to(device)
        else:
            x = x_generated.to(device)
        xs_pred, mu, logvar = model(x,s)
        #CUDA
        xs_pred = torch.nan_to_num(xs_pred)
        # #Solved?
        rec_loss = torch.mean(loss_rec(xs_pred, x))
        kl_loss = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + prior_weight*kl_loss
        train_loss += float(loss)
        opti.zero_grad()
        loss.backward()
        opti.step()
    scheduler.step()
    train_loss /= len(data_loader)
    if verbose:
        print(f"Train loss:\t {train_loss:.5f}")#| Train accuracy: {train_acc:.2f}%")
    return train_loss


# -

def val_cvae(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module,
            devide: torch.device = device, verbose:bool=True):
    model.to(device)
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for batch, (x,s,_) in enumerate(data_loader):
            x, s = x.to(device), s.to(device)
            xs_pred, _, _ = model(x, s)
            xs_pred = torch.nan_to_num(xs_pred)
            val_loss += loss_fn(xs_pred, x)
    val_temp = val_loss.detach()
    val_temp /= len(data_loader)
    if verbose:
        print(f'Test loss (CVAE): \t {val_temp:.5f}')
    return val_temp.item()


def combined_loss(predicted_space, true_space):
    #distance
    # One big loss or include multiple losses?
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    dist = 1-cos(predicted_space, true_space)
    return dist


def train_combined_diff(vae_model, gen_model, cvae_model, data_loader: torch.utils.data.DataLoader,
                        ab_t, a_t, b_t, ts,
                        opti: torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler,
                        loss_fn = combined_loss, device: torch.device=device,
                        verbose:bool=False, hyperbolic=False):
        
        #Total loss or indicidual component loss?

        vae_model.to(device).train()
        gen_model.to(device).train()
        cvae_model.to(device).train()
        
        context_layer = []
        for i in vae_model.encoder.encoder_stack:#[::-1]:
            if isinstance(i, nn.Linear):
                context_layer.append(i)
        #context_layer = context_layer.reverse()
        train_loss = 0.0
        
        for b, (x,s,_) in enumerate(data_loader):
            
            s = s.to(device)
            x = x.to(device)
            noise = torch.randn_like(x)
            
            # Semantic embedding
            c = []
            for l in range(1,len(context_layer)+1):
                temp_context_module = nn.ModuleList(context_layer[:l])
                s_context = s
                for m in temp_context_module:
                    s_context = m(s_context)
                c.append(s_context)
                
            #emb_space = vae_model(s)[0]
            
            # Generative from sampling by contex 
            #samples = torch.randn_like(x)
            #for i in range(ts,0,-1):
            #    t = torch.tensor([i/ts], device=device)[:,None]
            #    z = torch.randn_like(x) if i > 1 else 0
            #    eps = gen_model(samples, t, c)
            #    #x_pert = dm.perturb_input(x,t[i],noise, ab_t)
            #    samples = dm.denoise_add_noise(samples, i, eps, ab_t, a_t, b_t, z)
            #gen_space = samples
            gen_space = dm.sample_ddim(gen_model, x.shape[0], c, ab_t, ts)
            
            # Latent alignment
            s_pred, s_mu, s_var = cvae_model(gen_space,s)
            
            loss = loss_fn(s_pred,s)
            train_loss += float(loss.mean())
            opti.zero_grad()
            loss.mean().backward()
            opti.step()
        scheduler.step()
        train_loss /= len(data_loader)
        if verbose:
            print(f'Combined training loss: {train_loss:.4f}')
        return train_loss


# + jupyter={"outputs_hidden": true}
def train_combined(vae_model, gen_model, cvae_model, data_loader: torch.utils.data.DataLoader,
                   opti: torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler, 
                   loss_fn = combined_loss, device: torch.device=device,
                   verbose:bool=False, hyperbolic=False):
        

        vae_model.to(device).train()
        gen_model.to(device).train()
        cvae_model.to(device).train()
        
        train_loss = 0.0
        
        for b, (_,s,_) in enumerate(data_loader):
            s = s.to(device)
            s_pred, s_mu, s_var = cvae_model(gen_model(vae_model(s)[0]),s)
            loss = loss_fn(s_pred,s)
            train_loss += loss.mean()
            opti.zero_grad()
            loss.mean().backward()
            opti.step()
        scheduler.step()
        train_loss /= len(data_loader)
        if verbose:
            print(f'Combined training loss: {train_loss:.4f}')
        return train_loss.item()


# -

def train_combined_cond(vae_model, gen_model, cvae_model, data_loader: torch.utils.data.DataLoader,
                   opti: torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler, 
                   loss_fn, device: torch.device=device,
                   verbose:bool=False, hyperbolic=False):
        

        vae_model.to(device).train()
        gen_model.to(device).train()
        cvae_model.to(device).train()
        
        train_loss = 0.0
        
        for b, (_,s,_) in enumerate(data_loader):
            s = s.to(device)
            s_rep, s_mu, s_var = vae_model(s)
            z_cond = torch.normal(s_mu, s_var)
            gen_pred = gen_model(z_cond)
            s_pred = cvae_model(gen_pred, s)[0]
            loss = loss_fn(s_pred,s)
            train_loss += loss.mean()
            opti.zero_grad()
            loss.mean().backward()
            opti.step()
        scheduler.step()
        train_loss /= len(data_loader)
        if verbose:
            print(f'Combined training loss: {train_loss:.4f}')
        return train_loss.item()


def train_combined_cls(vae_model, gen_model, cvae_model, data_loader: torch.utils.data.DataLoader,
                   opti: torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler, cls_model,
                   loss_fn = combined_loss, device: torch.device=device,
                   verbose:bool=False):
        

        vae_model.to(device).train()
        gen_model.to(device).train()
        cvae_model.to(device).train()
        
        train_loss = 0.0
        
        for b, (x,s,y) in enumerate(data_loader):
            s = s.to(device)
            x = x.to(device)
            s_vae = vae_model(s)[0]
            x_gen = gen_model(s_vae)
            s_pred, s_mu, s_var = cvae_model(x_gen,s)
            
            x_gen = torch.nan_to_num(x_gen)
            loss_rec = torch.mean(loss_fn(s_pred,s))
            cls_pred = cls_model.predict_log_proba(x_gen.cpu().detach().numpy())
            cls_loss = np.mean([cls_pred[i][np.argmax(cls_pred[i])] for i in range(x_gen.shape[0])])
            
            loss = loss_rec + cls_loss
            train_loss += loss
            
            opti.zero_grad()
            loss.mean().backward()
            opti.step()
        scheduler.step()
        train_loss /= len(data_loader)
        if verbose:
            print(f'Combined training loss: {train_loss:.4f}')
        return train_loss.item()


def classification_space(s, semantic_enc, visual_gen,
                         conditional_enc, device=device):
    
    _aligned = False
    
    semantic_enc.to(device).eval()
    visual_gen.to(device).eval()
    conditional_enc.to(device).eval()
    
    with torch.no_grad():
        s_latent,_,_ = semantic_enc.encoder(s)
        x_gen = visual_gen(s_latent)
        if _aligned:
            s_rec = semantic_enc(s)[0]
            classifying_space,_,_ = conditional_enc.encoder(x_gen, s_rec)
        else:
            classifying_space,_,_ = conditional_enc.encoder(x_gen, s)
    
    return classifying_space


def unseen_accuracy(s_enc:nn.Module, vis_gen:nn.Module,cls_enc:nn.Module,
                    S_true, Y_true, truth, class_labels, device=device):
    
    #model=cvae.encoder, X_unseen=Xte, S_unseen=S_te_all, 
    #                S_true=S_te, Y_true=Y_te, true_cl_id=te_cl_id
    #model = cvae.encoder()
    #Transductive testing - the semantic space is known at testing time (paired with image)
    #TODO: Inductive testing - classifying in the image space instead (decode.dim = img.dim)
    
    #Predict        
    classifying_space = classification_space(S_true, s_enc, vis_gen, cls_enc, device)
    #classifying_space = classifying_space.detach().cpu()
    n_sample = classifying_space.shape[0]
                
    # Count correct
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred_lst = []
    acc = 0
    for i in range(n_sample):
        dist = 1-cos(classifying_space[i].unsqueeze(dim=0), truth)
        temp_pred = class_labels[np.argmin(dist.detach().cpu().numpy())]
        pred_lst.append(temp_pred)
        if (temp_pred == Y_true[i]).all():
            acc += 1
    
    return acc/n_sample


def distance_CS(pred_space, true_space, true_lbl, seen_lbls, gen_lbls, lamb, 
                device=device, cs=True):
    
    #Return distance matrix of dim: 1xK where K = number of true classes
    #pred space: predicted space of class: 0xD
    #true spae: true semantic space of all classes: KxD where D = number of features
    #true label: the true label of sample
    #seen label: list of seen labels 
             
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = 1-cos(pred_space.unsqueeze(dim=0), true_space)
    if (cs==True) and (true_lbl in seen_lbls):
        true_indx = int(np.where(true_lbl.detach().cpu().numpy() == gen_lbls.detach().cpu().numpy())[0])
        distance[true_indx] += lamb
    return distance


def generalized_accuracy(s_enc:nn.Module, vis_gen:nn.Module,cls_enc:nn.Module,
                         S_true, Y_true, truth, class_labels, class_lables_seen, 
                         lamb, device=device):
    
    #For generalized accuracy -> splitted on seen and unseen
    #S_true: predicted classification space
    #truth: true semantic space (for nearest neighbour)
    #class_labels: the labels of classes
    #class_labels_seen: the lables of seen classes only (a subset of all the classes)
    
    #Predict        
    classifying_space = classification_space(S_true, s_enc, vis_gen, cls_enc, device)
    #classifying_space = classifying_space.detach().cpu()
    n_sample = classifying_space.shape[0]
                
    # Count correct
    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred_lst = []
    acc, acc_seen, acc_unseen, cnt_unseen, cnt_seen = 0,0,0,0,0
    for i in range(n_sample):
        #dist = 1-cos(classifying_space[i].unsqueeze(dim=0), truth)
        dist = distance_CS(classifying_space[i], truth, Y_true[i], class_lables_seen, class_labels, lamb)
        temp_pred = class_labels[np.argmin(dist.detach().cpu().numpy())]
        pred_lst.append(temp_pred)
        if Y_true[i] in class_lables_seen:
            cnt_seen+=1
            if temp_pred==Y_true[i]:
                acc_seen+=1
                acc+=1
        else:
            cnt_unseen+=1
            if temp_pred==Y_true[i]:
                acc_unseen+=1
                acc+=1
    acc_seen/=cnt_seen
    acc_unseen/=cnt_unseen
    acc/=n_sample
    
    #return (2*acc_unseen*acc_seen)/(acc_unseen+acc_seen), acc
    return acc_seen, acc_unseen


def gen_sample_latent(vae_model, gen_model, s, n_sample, ts, ab_t, a_t, b_t, img_dim = 2048):
    context_layer = []
    for i in vae_model.encoder.encoder_stack:#[::-1]:
        if isinstance(i, nn.Linear):
            context_layer.append(i)
    
    c = []
    for l in range(1,len(context_layer)+1):
        temp_context_module = nn.ModuleList(context_layer[:l])
        s_context = s
        for m in temp_context_module:
            s_context = m(s_context)
        c.append(s_context)
                
    
    samples = torch.randn(n_sample, img_dim).to(device)
    for i in range(ts,0,-1):
        t = torch.tensor([i/ts], device=device)[:,None]
        z = torch.randn(n_sample, img_dim).to(device) if i > 1 else 0
        eps = gen_model(samples, t, c)
        #x_pert = dm.perturb_input(x,t[i],noise, ab_t)
        samples = dm.denoise_add_noise(samples, i, eps, ab_t, a_t, b_t, z)
    return samples


def unseen_accuracy_diff(s_enc:nn.Module, vis_gen, cls_enc:nn.Module,
                         ts, ab_t, a_t, b_t,
                         S_true, Y_true, truth, class_labels, class_lables_seen, lamb,
                         device=device):

    #Predict
    s_enc.to(device).eval()
    vis_gen.to(device).eval()
    cls_enc.to(device).eval()
    
    #s_latent = s_enc(S_true)[0]
    with torch.no_grad():
        x_gen = gen_sample_latent(s_enc, vis_gen, S_true, S_true.shape[0], ts, ab_t, a_t, b_t)
        classifying_space,_,_ = cls_enc.encoder(x_gen, S_true)
    
    n_sample = classifying_space.shape[0]
    truth = torch.unique(S_true, dim=0)
                
    pred_lst = []
    acc, acc_seen, acc_unseen, cnt_unseen, cnt_seen = 0,0,0,0,0
    for i in range(n_sample):
        #dist = 1-cos(classifying_space[i].unsqueeze(dim=0), truth)
        dist = distance_CS(classifying_space[i], truth, Y_true[i], class_lables_seen, class_labels, lamb)
        temp_pred = class_labels[np.argmin(dist.detach().cpu().numpy())]
        pred_lst.append(temp_pred)
        if Y_true[i] in class_lables_seen:
            cnt_seen+=1
            if temp_pred==Y_true[i]:
                acc_seen+=1
                acc+=1
        else:
            cnt_unseen+=1
            if temp_pred==Y_true[i]:
                acc_unseen+=1
                acc+=1
    acc_seen/=cnt_seen
    acc_unseen/=cnt_unseen
    acc/=n_sample
    
    return [acc_seen, acc_unseen]


def weights_init(m, normal=False):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        if normal:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        else:
            nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def save_array(lst, name, save_path=os.path.expanduser('~') + '/data/gan_vae_results/'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    temp = np.array(lst)
    np.save(save_path + name + '.npy', temp)


def save_model_weights(model, name, filepath=os.path.expanduser('~') + '/data/gan_vae_model/'):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    torch.save(model.state_dict(), filepath + name + '.pth')
    #print(f"Model weights saved to '{filepath}'.")


def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    import seaborn as sns

    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))
    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(
            val_dict[key],
            ax=key_ax,
            color=color,
            bins=50,
            stat=stat,
            kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
        )  # Only plot kde if there is variance
        hidden_dim_str = (
            r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape) > 1 else ""
        )
        key_ax.set_title(f"{key} {hidden_dim_str}")
        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1
    fig.subplots_adjust(wspace=0.4)
    return fig


def visualize_weight_distribution(model, color="C0"):
    weights = {}
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            continue
        key_name = f"Layer {name.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    # Plotting
    fig = plot_dists(weights, color=color, xlabel="Weight vals")
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.show()
    plt.close()


def triplet_same_semnatic(s, std=0.01):
    noise = (std**0.5)*torch.randn(s.shape)
    neg = torch.ones(s.shape)
    return s, s + noise, neg


def triplet_semantic(batch_s, bacth_y, same_semeantic=True):
    sem, lbl = batch_s, bacth_y
    least_two_samples = False
    while not least_two_samples:
        class_choice = random.choice(lbl)
        class_indices = torch.where(lbl == class_choice)[0]
        if (len(torch.unique(lbl)) == 1) or (len(class_indices)<3):
            #print('not uinique sample to create triplet')
            return triplet_same_semnatic(sem)
        else:
            a_index = random.choice(class_indices) 
            n_index = torch.where(lbl != class_choice)[0]
            p_index = class_indices[torch.where(class_indices != a_index)[0]]

            return torch.unsqueeze(sem[a_index,],0), sem[p_index], sem[n_index]


def train_vae_triplet(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
                      loss_triplet: torch.nn.Module,
                      optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                      device: torch.device = device, verbose:bool=True):
    train_loss = 0
    model.to(device).train()
    for batch, (x,s,y) in enumerate(data_loader):
        # Send data to GPU
        s_anc, s_pos, s_neg = triplet_semantic(s, y)
        s_anc, s_pos, s_neg = s_anc.to(device), s_pos.to(device), s_neg.to(device)
        # 1. Forward pass
        # Using all example per batch... Other alternatives?
        s_train = torch.cat((s_anc, s_pos, s_neg),axis=0).to(device)
        s_pred, s_mu, s_var = model(s_train)
        # 2. Calculate loss
        # Using only one example to calculate triplet loss
        # Alternativly, repeat the anchor and postive to match size of negatives?
        loss_rec = torch.mean(loss_fn(s_pred, s_train))
        loss_trip = loss_triplet(s_anc, s_pred[0].unsqueeze(0), s_neg[0].unsqueeze(0))
        loss_kl = -0.5*torch.mean(1 + s_var - s_mu.pow(2) - s_var.exp())
        loss = loss_rec + loss_kl + loss_trip
        train_loss += loss
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
    # Calculate loss and accuracy per epoch and print out what's happening
    scheduler.step()
    train_loss /= len(data_loader)
    if verbose:
        print(f"Train loss: {train_loss:.5f}")#| Train accuracy: {train_acc:.2f}%")
    return train_loss.item()



def within_covariance(data, labels, device=device):
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    # Calculate class means
    class_means = []
    for label in unique_labels:
        class_data = data[labels == label]
        class_mean = torch.mean(class_data, dim=0)
        class_means.append(class_mean)
    
    class_means = torch.stack(class_means)

    # Initialize the within-class covariance matrix
    within_covariance_matrix = torch.zeros((data.shape[1], data.shape[1]), device=device)

    
    for i,label in enumerate(unique_labels):
        class_data = data[labels == label]
        num_samples = len(class_data)
        
        # Calculate the deviation of each data point from its class mean
        deviations = class_data - class_means[i]
        #print(deviations.shape)
        
        # Calculate the covariance matrix for the current class
        if num_samples != 1:
            covariance_matrix = torch.matmul(deviations.t(), deviations) / (num_samples - 1)
        else:
            covariance_matrix = torch.matmul(deviations.t(), deviations) / (1)
        #print(covariance_matrix.shape)
        
        # Add the covariance matrix to the within-class covariance matrix
        within_covariance_matrix += covariance_matrix
    
    return within_covariance_matrix


def between_covariance(data, labels, device=device):
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    # Calculate class means
    class_means = []
    class_sampels = []
    for label in unique_labels:
        class_data = data[labels == label]
        class_sampels.append(len(class_data))
        class_mean = torch.mean(class_data, dim=0)
        class_means.append(class_mean)
    
    class_means = torch.stack(class_means)
    
    # Calculate the overall mean
    overall_mean = torch.mean(class_means, dim=0)
    
    # Initialize the between-class covariance matrix
    between_covariance_matrix = torch.zeros((data.shape[1], data.shape[1]), device=device)
    
    for i,label in enumerate(unique_labels):
        class_mean_diff = class_means[i] - overall_mean
        
        # Calculate the covariance matrix for the current class
        covariance_matrix = torch.outer(class_mean_diff, class_mean_diff)
        
        # Add the covariance matrix to the between-class covariance matrix
        if class_sampels[i] ==1:
            between_covariance_matrix += covariance_matrix
        else:    
            between_covariance_matrix += covariance_matrix/(class_sampels[i]-1)
        #without nomralization:
        #between_covariance_matrix += covariance_matrix
        
    return between_covariance_matrix


def fisher_criterion(s,l, lambd=1.0, device=device):
    sb = between_covariance(s,l, device)
    sw = within_covariance(s,l, device)
    
    # https://math.stackexchange.com/questions/1821508/trace-of-matrix-exponential-closed-form-expression
    return torch.trace(torch.linalg.matrix_exp(sb))/torch.trace(torch.linalg.matrix_exp(sw))
    
    #return abs(torch.log(torch.trace(sb)))/abs(torch.log(torch.trace(sw)))


# +
def train_vae_fisher(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, fisher_fn,
                     device: torch.device = device, verbose:bool=True):
    train_loss = 0
    model.to(device).train()
    for batch, (x,s,y) in enumerate(data_loader):
        # Send data to GPU
        s,y = s.to(device), y.to(device)
        # 1. Forward pass
        s_pred, s_mu, s_var = model(s)
        # 2. Calculate loss 
        # CUDA ERROR HERE!!
        s_pred = torch.nan_to_num(s_pred)
        # #solved?
        loss_rec = torch.mean(loss_fn(s_pred, s))
        loss_kl = -0.5*torch.mean(1 + s_var - s_mu.pow(2) - s_var.exp())
        loss_fisher = fisher_fn(s_pred,y) #Automatically move loss back to gpu.
        loss = loss_rec + loss_kl - loss_fisher
        train_loss += loss
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
    # Calculate loss and accuracy per epoch and print out what's happening
    scheduler.step()
    train_loss /= len(data_loader)
    if verbose:
        print(f"Train loss: {train_loss:.5f}")#| Train accuracy: {train_acc:.2f}%")
    return train_loss.item()


# -

def precision_recall(s_enc:nn.Module, vis_gen:nn.Module,cls_enc:nn.Module,
                    X_unseen, S_unseen, S_true, Y_true, true_cl_id, 
                    device=device):
    """
    Transductive testing - the semantic space is known at testing time (paired with image)
    Inductive testing - classifying in the image space instead (decode.dim = img.dim)
    """
    #Predict
    n_sample, x_dim, s_dim = X_unseen.shape[0], X_unseen.shape[1], S_unseen.shape[1]
    test_img = torch.tensor(X_unseen,dtype=torch.float).to(device)
    test_semantic = torch.tensor(S_unseen, dtype=torch.float).to(device)
    
    classifying_space = classification_space(test_semantic, s_enc, vis_gen, cls_enc)

    test_img = test_img.detach().cpu()
    test_semantic = test_semantic.detach().cpu()
    classifying_space = classifying_space.detach().cpu()

    # Count correct
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    truth = torch.tensor(S_true, dtype=torch.float)
    correct_pred = 0
    wrong_pred = 0
    
    # Alt 1
    #for i in range(n_sample):
    #    dist = cos(classifying_space[i], truth)
    #    pred_label = true_cl_id[np.argmin(dist.detach().numpy())]
    #    true_label = Y_true[i]
    #    if pred_label == true_label:
    #        correct_pred += 1
    #    else:
    #        wrong_pred += 1
    #        
    #precision = correct_pred/(correct_pred+wrong_pred)
    #recall = correct_pred/(correct_pred+(correct_pred-wrong_pred))
    #
    #return precision, recall
                           
    # Alt 2
    pred_lst = []
    for i in range(n_sample):
        dist = cos(classifying_space[i], truth)
        pred_lst.append(true_cl_id[np.argmin(dist.detach().numpy())])
    
    confuse = confusion_matrix(truth, pred_lst)
    return confuse


def train_hvae(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module,
               loss_additional, curvatur,
               optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
               device: torch.device = device, verbose:bool=False):
    train_loss = 0.0
    model.to(device).train()
    for b, (x,s,y) in enumerate(data_loader):
        s = s.to(device)
        # Z_hyp: hyperbolic embeding, S_rec: reconstructed input space (euclidean)
        z_hyp, s_rec = model(s)
        # Loss rec: reconstruction loss
        loss_rec = torch.mean(loss_fn(s_rec, s))
        # Uncertainty: hyperbolic embedding distance to origin.
        # Maximize distance = minimize negative loss
        loss_distance = - torch.mean(loss_additional(z_hyp, curvatur))
        loss = loss_rec + loss_distance
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_loss /= len(data_loader)
    if verbose:
        print(f"\trec loss: {loss_rec:.4f} | dist(x,0): {loss_distance:.4f}")#| Train accuracy: {train_acc:.2f}%")
    return train_loss


def RankMe(space):
    _,singular_value,_ = torch.linalg.svd(space, full_matrices=False)
    singular_norm = torch.norm(singular_value, p=1)
    rankme = torch.sum(torch.special.entr(torch.div(singular_value,singular_norm)))
    return rankme


def NEsum(space):
    cov_space = torch.cov(space.T)
    eig = torch.linalg.eigvals(cov_space)
    #eig, _ = torch.linalg.eig(cov_space)
    nesum = torch.sum(torch.div(eig, eig[0]))
    return torch.real(nesum)


def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = torch.max(original)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def Geometry_Score(x1, x2):
    rlts1 = gs.rlts(x1, L_0=32, gamma=1.0/8, i_max=100, n=100)
    rlts2 = gs.rlts(x2, L_0=32, gamma=1.0/8, i_max=100, n=100)
    geo_score = gs.geom_score(rlts1, rlts2)
    return geo_score


def calculate_fid(act1, act2, eps=1e-6, device=device):
    mu1 = torch.atleast_1d(torch.mean(act1, dim=0).to(device))
    mu2 = torch.atleast_1d(torch.mean(act2, dim=0).to(device))
    sigma1 = torch.matmul((act1 - mu1).T, (act1 - mu1)) / (act1.size(0) - 1)
    sigma2 = torch.matmul((act2 - mu2).T, (act2 - mu2)) / (act2.size(0) - 1)

    #ssdiff = torch.sum((mu1 - mu2) ** 2.0)
    diff = mu1 - mu2
    covmean = torch.sqrt(torch.matmul(sigma1,sigma2))
    #covmean = torch.sqrt(torch.matmul(torch.linalg.matrix_power(sigma1, 0.5), sigma2).matmul(torch.linalg.matrix_power(sigma1, 0.5)))
    
    if not torch.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        #print(msg)
        offset = torch.eye(sigma1.shape[0], device=device) * eps
        temp = torch.matmul((sigma1 + offset),(sigma2 + offset))
        covmean = scipy.linalg.sqrtm(temp.cpu())
        
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
        
    tr_covmean = torch.trace(torch.tensor(covmean))

    #fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    #return fid
    return (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)


def prototype_distance(points_matrix, true_point):
    # Ensure points_matrix and true_point are PyTorch tensors
    #points_matrix = torch.tensor(points_matrix, dtype=torch.float32)
    #true_point = torch.tensor(true_point, dtype=torch.float32)
    
    # Calculate the Euclidean distance between each point in the matrix and the true point
    distances = torch.norm(points_matrix - true_point, dim=1)
    
    # Calculate the average distance
    avg_distance = torch.mean(distances)
    
    return avg_distance


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 0.0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_diffusion(model, data_loader: torch.utils.data.DataLoader, loss_fn, optimizer,
                    time_sched, time_step, pertub_fn,
                    device: torch.device=device):
    train_loss = 0.0
    model.to(device).train()
    for batch, (x,_,_) in enumerate(data_loader):
        x = x.to(device)
        noise = torch.randn_like(x)
        t = torch.randint(1, time_step+1, (x.shape[0],1)).to(device)
        x_pert = pertub_fn(x, t, noise, time_sched)
        pred_noise = model(x_pert, t/time_step)
        
        loss = loss_fn(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #ema.update(model)
        
        train_loss += loss
        
    return train_loss/len(data_loader)


def val_diffusion(model, data_loader: torch.utils.data.DataLoader, loss_fn,
                  ab_t, a_t, ts,
                  devide: torch.device = device, verbose:bool=True):
    model.to(device)
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for batch, (x,_,_) in enumerate(data_loader):
            x = x.to(device)
            x_samp = dm.sample_ddpm(model, x.shape[0], ab_t, a_t, ts, input_dim=2048).to(device)
            loss = loss_fn(x_samp, x)
            val_loss += loss
    val_temp = val_loss.detach()
    val_temp /= len(data_loader)
    if verbose:
        print(f'Test loss (Diff): \t {val_temp:.5f}')
    return val_temp


def load_data_split_og(X,S,Y, GENERALIZED, class_lbls_unseen, b_size, device=device):
    
    
    
            
    acc_mean = []
    for acc_split in test_loader:
        acc_mean.append(eg.unseen_accuracy(vae, generator, cvae,
                                 acc_split[1].to(device), acc_split[2].to(device),
                                 s_unseen_true, class_lbls_unseen, 
                                 device=device)
                       )
    acc = sum(acc_mean)/len(acc_mean)
    # Generalized with
    # RANDOM SPLIT
    gen = GENERALIZED
    
    GENERALIZED = gen
    random_split = rand

    X = np.concatenate([X_tr, X_te])
    S = np.concatenate([S_tr, S_te_all])
    Y = np.concatenate([Y_tr, Y_te])

    if random_split:
        class_lbls_unseen = np.array(random.sample(list(np.unique(Y)), 10))
    else:
        class_lbls_unseen = np.array(te_cl_id)

    #top_dist_awa = [6, 12,  5, 47, 49, 13, 46, 15, 19, 10]

    train_loader, test_loader, s_unseen_true, x_cls, y_cls = eg.load_data_split(X,S,Y,
                                                                                GENERALIZED,
                                                                                class_lbls_unseen,
                                                                                b_size)

    #_,test_sem, test_lab = zip(*[batch for batch in test_loader]) 
    #test_sem, test_lab = torch.cat(test_sem, dim=0).to(device), torch.cat(test_lab, dim=0).to(device)

    x_unseen, s_unseen, y_unseen = [],[], []
    x_seen, s_seen, y_seen = [],[], []
    for i in range(len(Y)):
        if Y[i] in class_lbls_unseen:
            x_unseen.append(X[i])
            s_unseen.append(S[i])
            y_unseen.append(Y[i])
        else:
            x_seen.append(X[i])
            s_seen.append(S[i])
            y_seen.append(Y[i])

    #x_unseen, s_unseen, y_unseen = np.array(x_unseen), np.array(s_unseen), np.array(y_unseen)
    #x_seen, s_seen, y_seen = np.array(x_seen), np.array(s_seen), np.array(y_seen)

    if gen:
        _,x_gen, _,s_gen,_,y_gen = train_test_split(x_seen, s_seen, y_seen, test_size=.2)
        x_unseen = np.concatenate([x_unseen, x_gen]) 
        s_unseen = np.concatenate([s_unseen, s_gen])
        y_unseen = np.concatenate([y_unseen, y_gen])
    
    x_unseen, s_unseen, y_unseen = np.array(x_unseen), np.array(s_unseen), np.array(y_unseen)
    x_seen, s_seen, y_seen = np.array(x_seen), np.array(s_seen), np.array(y_seen)
    
    x_unseen, s_unseen, y_unseen = torch.tensor(x_unseen), torch.tensor(s_unseen), torch.tensor(y_unseen)
    x_seen, s_seen, y_seen = torch.tensor(x_seen), torch.tensor(s_seen), torch.tensor(y_seen)

    scaler_X = preprocessing.MinMaxScaler()
    scaler_S = preprocessing.MinMaxScaler()


    X = scaler_X.fit_transform(x_seen)
    Xte = scaler_X.transform(x_unseen)
    S = scaler_S.fit_transform(s_seen)
    Ste = scaler_S.transform(s_unseen)

    #Seen
    train_data = mc.custom_dataset_lbl(X, S, y_seen)
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True)

    #val_data = mc.custom_dataset_lbl(Xte, Ste, np.asarray(y_val))
    #val_loader = DataLoader(val_data, batch_size=b_size, shuffle=True)

    #Unseen
    test_data = mc.custom_dataset_lbl(Xte, Ste, y_unseen)
    test_loader = DataLoader(test_data, batch_size=b_size)

    s_unseen_true = []
    for i in class_lbls_unseen:
        for j in range(len(y_unseen)):
            if y_unseen[j] == i:
                s_unseen_true.append(torch.tensor(Ste[j], dtype=torch.float))
                break

    s_unseen_true = torch.stack(s_unseen_true).to(device)
    class_lbls_unseen = torch.tensor(class_lbls_unseen, device=device)
    s_unseen = s_unseen.to(torch.float).to(device)
    y_unseen = y_unseen.to(device)

    return train_loader, test_loader, s_unseen_true, X, y_seen


def load_len_data(df_name, generalized, split, b_size, src='OG', device=device):
    
    df_folder = os.path.expanduser('~') + '/data/EnrichedSemanticEmbedding/'
        
    if df_name.lower() == 'awa':
        X_tr, X_te, S_tr, S_te, Y_te, te_cl_id, Y_tr, S_te_all = ld.awa2(norm_data=False, int_proj=False, src=src)
        S_tr = np.load(df_folder + 'trainDataAttrs_AWA2.npy')
        S_te_all = np.load(df_folder + 'testDataAttrs_AWA2.npy')
        
    elif df_name.lower() == 'cub':
        X_tr, X_te, S_tr, S_te, Y_te, te_cl_id, Y_tr, S_te_all = ld.cub(norm_data=False, int_proj=False, src=src)
        S_tr = np.load(df_folder + 'trainDataAttrs_CUB.npy')
        S_te_all = np.load(df_folder + 'testDataAttrs_CUB.npy')
    elif df_name.lower() == 'sun':
        X_tr, X_te, S_tr, S_te, Y_te, te_cl_id, Y_tr, S_te_all = ld.sun(norm_data=False, int_proj=False, src=src)
        S_tr = np.load(df_folder + 'trainDataAttrs_SUN.npy')
        S_te_all = np.load(df_folder + 'testDataAttrs_SUN.npy')
    else:
        print('Not valid dataset')
        return None
    
    GENERALIZED = generalized
    random_split = split

    X = np.concatenate([X_tr, X_te])
    S = np.concatenate([S_tr, S_te_all])
    Y = np.concatenate([Y_tr, Y_te])

    class_lbls = list(np.unique(Y))
    if random_split:
        class_lbls_unseen = np.array(random.sample(class_lbls, 10))
    else:
        class_lbls_unseen = np.array(te_cl_id)

    class_lbls_seen = np.array(list(set(class_lbls) - set(class_lbls_unseen)))

    if GENERALIZED:
        class_lbls_gen = np.array(random.sample(list(class_lbls_seen), int(len(class_lbls_seen)*0.2)))
        class_lbls_gen = np.concatenate([class_lbls_unseen, class_lbls_gen])


    x_unseen, s_unseen, y_unseen = [],[], []
    x_seen, s_seen, y_seen = [],[], []
    for i in range(len(Y)):
        if Y[i] in class_lbls_seen:
            x_seen.append(X[i])
            s_seen.append(S[i])
            y_seen.append(Y[i])
        else:
            x_unseen.append(X[i])
            s_unseen.append(S[i])
            y_unseen.append(Y[i])


    x_unseen, s_unseen, y_unseen = np.array(x_unseen), np.array(s_unseen), np.array(y_unseen)
    x_seen, s_seen, y_seen = np.array(x_seen), np.array(s_seen), np.array(y_seen)

    x_unseen, s_unseen, y_unseen = torch.tensor(x_unseen), torch.tensor(s_unseen), torch.tensor(y_unseen)
    x_seen, s_seen, y_seen = torch.tensor(x_seen), torch.tensor(s_seen), torch.tensor(y_seen)


    scaler_X = preprocessing.MinMaxScaler()
    scaler_S = preprocessing.MinMaxScaler()


    Xtr = scaler_X.fit_transform(x_seen)
    Xte = scaler_X.transform(x_unseen)
    Str = scaler_S.fit_transform(s_seen)
    Ste = scaler_S.transform(s_unseen)

    #Seen
    train_data = mc.custom_dataset_lbl(Xtr, Str, y_seen)
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True)

    #Unseen
    test_data = mc.custom_dataset_lbl(Xte, Ste, y_unseen)
    test_loader = DataLoader(test_data, batch_size=b_size)

    s_unseen_true = []
    for i in class_lbls_unseen:
        temp_i = np.where(y_unseen == i)[0][0] #first occerence, all are the same for semantic space
        s_unseen_true.append(torch.tensor(Ste[temp_i], dtype=torch.float))
    s_unseen_true = torch.stack(s_unseen_true).to(device)

    if GENERALIZED:  
        s_unseen_gen = []
        s_gen, y_gen = [],[]
        for ii in class_lbls_gen:
            if ii in class_lbls_seen:
                temp_ii = np.where(y_seen == ii)[0] #first occerence, all are the same for semantic space
                
                s_unseen_gen.append(torch.tensor(Str[temp_ii[0]], dtype=torch.float))
                s_gen.append(torch.tensor(Str[temp_ii], dtype=torch.float))
                y_gen.append(y_seen[temp_ii])

            elif ii in class_lbls_unseen:
                temp_ii = np.where(y_unseen == ii)[0]
                
                s_unseen_gen.append(torch.tensor(Ste[temp_ii[0]], dtype=torch.float))
                s_gen.append(torch.tensor(Ste[temp_ii], dtype=torch.float))
                y_gen.append(y_unseen[temp_ii])
            else:
                print('fatal error in dataset-split. Stopping!')
                return None


        s_unseen_gen = torch.stack(s_unseen_gen).to(device)
        s_gen = torch.cat(s_gen, dim=0).to(device)
        y_gen = torch.cat(y_gen).to(device)
                
            
    s_seen_true = []
    for iii in class_lbls_seen:
        temp_iii = np.where(y_seen == iii)[0][0]
        s_seen_true.append(torch.tensor(Str[temp_iii], dtype=torch.float))
    s_seen_true = torch.stack(s_seen_true).to(device)


    class_lbls_unseen = torch.tensor(class_lbls_unseen, device=device)
    if GENERALIZED:
        class_lbls_gen = torch.tensor(class_lbls_gen, device=device)
    class_lbls_seen = torch.tensor(class_lbls_seen, device=device)
    
    Str = torch.tensor(Str, dtype=torch.float, device=device)
    Ste = torch.tensor(Ste, dtype=torch.float, device=device)
    y_unseen = y_unseen.to(device)
    y_seen = y_seen.to(device)
    
    if GENERALIZED:
        return [train_loader, test_loader], [s_seen_true, s_unseen_true, s_unseen_gen], [class_lbls_seen,class_lbls_unseen,class_lbls_gen], [Str, Ste, s_gen], [y_seen, y_unseen, y_gen], Xtr
    else:
        return [train_loader, test_loader], [s_seen_true, s_unseen_true], [class_lbls_seen,class_lbls_unseen], [Str, Ste], [y_seen, y_unseen], Xtr


def load_data_split(df_name, generalized, split, b_size, src='OG', device=device):
    
    if df_name.lower() == 'awa':
        X_tr, X_te, S_tr, S_te, Y_te, te_cl_id, Y_tr, S_te_all = ld.awa2(norm_data=False, int_proj=False, src=src)
    elif df_name.lower() == 'cub':
        X_tr, X_te, S_tr, S_te, Y_te, te_cl_id, Y_tr, S_te_all = ld.cub(norm_data=False, int_proj=False, src=src)
    elif df_name.lower() == 'sun':
        X_tr, X_te, S_tr, S_te, Y_te, te_cl_id, Y_tr, S_te_all = ld.sun(norm_data=False, int_proj=False, src=src)
    else:
        print('Not valid dataset')
        return None
    
    GENERALIZED = generalized
    random_split = split

    X = np.concatenate([X_tr, X_te])
    S = np.concatenate([S_tr, S_te_all])
    Y = np.concatenate([Y_tr, Y_te])

    class_lbls = list(np.unique(Y))
    if random_split:
        class_lbls_unseen = np.array(random.sample(class_lbls, 10))
    else:
        class_lbls_unseen = np.array(te_cl_id)

    class_lbls_seen = np.array(list(set(class_lbls) - set(class_lbls_unseen)))

    if GENERALIZED:
        class_lbls_gen = np.array(random.sample(list(class_lbls_seen), int(len(class_lbls_seen)*0.2)))
        class_lbls_gen = np.concatenate([class_lbls_unseen, class_lbls_gen])


    x_unseen, s_unseen, y_unseen = [],[], []
    x_seen, s_seen, y_seen = [],[], []
    for i in range(len(Y)):
        if Y[i] in class_lbls_seen:
            x_seen.append(X[i])
            s_seen.append(S[i])
            y_seen.append(Y[i])
        else:
            x_unseen.append(X[i])
            s_unseen.append(S[i])
            y_unseen.append(Y[i])


    x_unseen, s_unseen, y_unseen = np.array(x_unseen), np.array(s_unseen), np.array(y_unseen)
    x_seen, s_seen, y_seen = np.array(x_seen), np.array(s_seen), np.array(y_seen)

    x_unseen, s_unseen, y_unseen = torch.tensor(x_unseen), torch.tensor(s_unseen), torch.tensor(y_unseen)
    x_seen, s_seen, y_seen = torch.tensor(x_seen), torch.tensor(s_seen), torch.tensor(y_seen)


    scaler_X = preprocessing.MinMaxScaler()
    scaler_S = preprocessing.MinMaxScaler()


    Xtr = scaler_X.fit_transform(x_seen)
    Xte = scaler_X.transform(x_unseen)
    Str = scaler_S.fit_transform(s_seen)
    Ste = scaler_S.transform(s_unseen)

    #Seen
    train_data = mc.custom_dataset_lbl(Xtr, Str, y_seen)
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True)

    #Unseen
    test_data = mc.custom_dataset_lbl(Xte, Ste, y_unseen)
    test_loader = DataLoader(test_data, batch_size=b_size)

    s_unseen_true = []
    for i in class_lbls_unseen:
        temp_i = np.where(y_unseen == i)[0][0] #first occerence, all are the same for semantic space
        s_unseen_true.append(torch.tensor(Ste[temp_i], dtype=torch.float))
    s_unseen_true = torch.stack(s_unseen_true).to(device)

    if GENERALIZED:  
        s_unseen_gen = []
        s_gen, y_gen = [],[]
        for ii in class_lbls_gen:
            if ii in class_lbls_seen:
                temp_ii = np.where(y_seen == ii)[0] #first occerence, all are the same for semantic space
                
                s_unseen_gen.append(torch.tensor(Str[temp_ii[0]], dtype=torch.float))
                s_gen.append(torch.tensor(Str[temp_ii], dtype=torch.float))
                y_gen.append(y_seen[temp_ii])

            elif ii in class_lbls_unseen:
                temp_ii = np.where(y_unseen == ii)[0]
                
                s_unseen_gen.append(torch.tensor(Ste[temp_ii[0]], dtype=torch.float))
                s_gen.append(torch.tensor(Ste[temp_ii], dtype=torch.float))
                y_gen.append(y_unseen[temp_ii])
            else:
                print('fatal error in dataset-split. Stopping!')
                return None


        s_unseen_gen = torch.stack(s_unseen_gen).to(device)
        s_gen = torch.cat(s_gen, dim=0).to(device)
        y_gen = torch.cat(y_gen).to(device)
                
            
    s_seen_true = []
    for iii in class_lbls_seen:
        temp_iii = np.where(y_seen == iii)[0][0]
        s_seen_true.append(torch.tensor(Str[temp_iii], dtype=torch.float))
    s_seen_true = torch.stack(s_seen_true).to(device)


    class_lbls_unseen = torch.tensor(class_lbls_unseen, device=device)
    if GENERALIZED:
        class_lbls_gen = torch.tensor(class_lbls_gen, device=device)
    class_lbls_seen = torch.tensor(class_lbls_seen, device=device)
    
    Str = torch.tensor(Str, dtype=torch.float, device=device)
    Ste = torch.tensor(Ste, dtype=torch.float, device=device)
    y_unseen = y_unseen.to(device)
    y_seen = y_seen.to(device)
    
    if GENERALIZED:
        return [train_loader, test_loader], [s_seen_true, s_unseen_true, s_unseen_gen], [class_lbls_seen,class_lbls_unseen,class_lbls_gen], [Str, Ste, s_gen], [y_seen, y_unseen, y_gen], Xtr
    else:
        return [train_loader, test_loader], [s_seen_true, s_unseen_true], [class_lbls_seen,class_lbls_unseen], [Str, Ste], [y_seen, y_unseen], Xtr

