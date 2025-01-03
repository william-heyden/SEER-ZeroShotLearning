
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy
import random



class vae_simple_dense(nn.Module):
    """
    For pre-extracted features of AWA, CUB, and SUN. 
    hidden_layers: dimension of the linear layers [l1>l2>...>ln] s.t. ln>laten_dim
    """
    def __init__(self, hidden_layers, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = len(hidden_layers)
        
        enc = []
        for l in hidden_layers:
            enc.append(
                nn.Sequential(
                    nn.Linear(in_features=self.input_dim, out_features=l),
                    nn.BatchNorm1d(l),
                    nn.ReLU()
                )
            )
            self.input_dim = l
        self.encode = nn.Sequential(*enc)
        self.encode_mu = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        self.encode_var = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        
        dec = []
        for i in range(1,self.num_layers):
            dec.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[self.num_layers-i], out_features=hidden_layers[self.num_layers-i-1]),
                    nn.BatchNorm1d(hidden_layers[self.num_layers-i-1]),
                    nn.ReLU()
                )
            )
        #Add sigmoid output last layer
        dec.append(nn.Sequential(
            nn.Linear(in_features=hidden_layers[0], out_features=self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Sigmoid()
            )
        )
        self.decode=nn.Sequential(*dec)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, data):
        #return self.encode(data)
        z = self.encode(data)
        z_mu, z_std = self.encode_mu(z), self.encode_var(z)
        return self.reparameterize(z_mu, z_std), z_mu, z_std
        
    
    def decoder(self, latent_data):
        return self.decode(latent_data)
    
    def forward(self,x):
        z, z_mu, z_var = self.encoder(x)
        return self.decoder(z), z_mu, z_var
            

class vae_encoder(nn.Module):
    def __init__(self, hidden_layer, input_dim, latent_dim, p):
        super().__init__()
        self.latent = latent_dim
        self.input_d = input_dim
        self.p = p
        
        self.encoder_stack = nn.Sequential(
            nn.Linear(in_features=self.input_d, out_features=hidden_layer[0]),
            nn.BatchNorm1d(hidden_layer[0]),
            nn.ReLU(),
            nn.Dropout(self.p),
            
            nn.Linear(in_features=hidden_layer[0], out_features=hidden_layer[1]),
            nn.BatchNorm1d(hidden_layer[1]),
            nn.ReLU(),
            nn.Dropout(self.p),
            
            nn.Linear(in_features=hidden_layer[1], out_features=hidden_layer[2]),
            nn.BatchNorm1d(hidden_layer[2]),
            nn.ReLU(),
            nn.Dropout(self.p),
        )
        self.encoder_mu = nn.Linear(in_features=hidden_layer[2], out_features=self.latent)
        self.encoder_var = nn.Linear(in_features=hidden_layer[2], out_features=self.latent)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        z = self.encoder_stack(x)
        z_mu, z_std = self.encoder_mu(z), self.encoder_var(z)
        return self.reparameterize(z_mu, z_std), z_mu, z_std


class vae_decoder(nn.Module):
    def __init__(self, hidden_layer, input_dim, latent_dim, p):
        super().__init__()
        self.latent = latent_dim
        self.input_d = input_dim
        self.p = p
        
        self.decoder_stack = nn.Sequential(
            nn.Linear(in_features=self.latent, out_features=hidden_layer[2]),
            nn.BatchNorm1d(hidden_layer[2]),
            nn.ReLU(),
            nn.Dropout(self.p),
            
            nn.Linear(in_features=hidden_layer[2], out_features=hidden_layer[1]),
            nn.BatchNorm1d(hidden_layer[1]),
            nn.ReLU(),
            nn.Dropout(self.p),
            
            nn.Linear(in_features=hidden_layer[1], out_features=hidden_layer[0]),
            nn.BatchNorm1d(hidden_layer[0]),
            nn.ReLU(),
            nn.Dropout(self.p),
            
            nn.Linear(in_features=hidden_layer[0], out_features=self.input_d),
            nn.BatchNorm1d(self.input_d),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder_stack(x)


class vae_simple(nn.Module):
    def __init__(self, hidden_layers, input_dim, latent_dim, p):
        super().__init__()
        self.encoder= vae_encoder(hidden_layers, input_dim, latent_dim, p)
        self.decoder=vae_decoder(hidden_layers, input_dim, latent_dim, p)
        
    def forward(self, x):
        z, zm, zs = self.encoder(x)
        return self.decoder(z), zm, zs
    
    def loss_function(self, recon_x, x, mu, logvar, kl_weight):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_KLD = KLD.sum()
        return BCE + (kl_weight*total_KLD), BCE, total_KLD, KLD


class vae_simple_encoder(nn.Module):
    def __init__(self, hidden_layers, input_dim, latent_dim):
        super().__init__()
        #self.input_dim = input_dim
        self.latent_dim = latent_dim
        #self.num_layers = len(hidden_layers)
        temp_dim = input_dim
        enc = []
        for l in hidden_layers:
            enc.append(
                nn.Sequential(
                    nn.Linear(in_features=temp_dim, out_features=l),
                    nn.BatchNorm1d(l),
                    nn.ReLU()
                )
            )
            temp_dim = l
        self.encode = nn.Sequential(*enc)
        self.encode_mu = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        self.encode_var = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        z = self.encode(x)
        z_mu, z_std = self.encode_mu(z), self.encode_var(z)
        return self.reparameterize(z_mu, z_std), z_mu, z_std


class vae_simple_decoder(nn.Module):
    def __init__(self, hidden_layers, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = len(hidden_layers)
        
        dec = []
        dec.append(nn.Linear(in_features=self.latent_dim, out_features=hidden_layers[-1]))
        for i in range(1,self.num_layers):
            dec.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[self.num_layers-i], out_features=hidden_layers[self.num_layers-i-1]),
                    nn.BatchNorm1d(hidden_layers[self.num_layers-i-1]),
                    nn.ReLU()
                )
            )
        #Add sigmoid output last layer
        dec.append(nn.Sequential(
            nn.Linear(in_features=hidden_layers[0], out_features=self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Sigmoid()
            )
        )
        self.decode=nn.Sequential(*dec)
    
    def forward(self, x):
        return self.decode(x)


class vae_simple_cnn(nn.Module):
    def __init__(self, hidden_layers_dim, input_dim, latent_dim):
        super().__init__()
        self.dec_out = input_dim
        self.latent_dim = latent_dim
        
        #ENCODER
        enc = []   
        for l in hidden_layers_dim:
            enc.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=input_dim, out_channels=l, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(l),
                    nn.ReLU()
                    )
            )
            input_dim = l
        self.encoder = nn.Sequential(*enc)
        
        self.encoder_mu = nn.Linear(in_features=hidden_layers_dim[-1], out_features=self.latent_dim)
        self.encoder_std = nn.Linear(in_features=hidden_layers_dim[-1], out_features=self.latent_dim)
        
        #DECODER
        self.dec_input = nn.Linear(in_features=self.latent_dim, out_features=hidden_layers_dim[-1])
        dec = []
        for i,l in reversed(list(enumerate(hidden_layers_dim))):
            if i != 0:
                out_chnl = hidden_layers_dim[i-1]
            else:
                out_chnl = self.dec_out
            dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=l, out_channels=out_chnl, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_chnl),
                    nn.ReLU()
                )
            )
        self.decoder = nn.Sequential(*dec)
        
    def repara(self,m,s):
        std = torch.exp(0.5 * s)
        eps = torch.randn_like(std)
        
        return eps * std + m

    
    def forward(self, x):
        z = self.encoder(x)
        enc_mu = self.encoder_mu(z)
        enc_std = self.encoder_std(z)
        enc_out = self.repara(enc_mu, enc_out)
        
        dec_in = self.dec_input(enc_out)
        return self.decoder(dec_in)




# GENERATOR
class generator_simple_dense(nn.Module):
    def __init__(self, gen_layers:list, z_shape:int, img_shape:int, p):
        super().__init__()
        self.gen_layers = gen_layers
        self.gen_input = z_shape
        self.img_shape = img_shape
        self.p = p
        
        gen = []
        for l in gen_layers:
            gen.append(nn.Sequential(
                nn.Linear(in_features=self.gen_input, out_features=l),
                nn.BatchNorm1d(l),
                nn.ReLU(),
                nn.Dropout(self.p)
                )
            )
            self.gen_input = l
            
        gen.append(nn.Sequential(
            nn.Linear(in_features=self.gen_input, out_features=self.img_shape),
            #nn.ReLU()
            #nn.Sigmoid()
            )
        )
                   
        self.generat = nn.Sequential(*gen)

    def forward(self, x):
        return self.generat(x)
                   
class c_generator_simple_dense(nn.Module):
    def __init__(self, gen_layers:list, z_shape:int, img_shape:int, p):
        super().__init__()
        self.gen_layers = gen_layers
        self.gen_input = z_shape + 1
        self.img_shape = img_shape
        self.p = p
        
        gen = []
        for l in gen_layers:
            gen.append(nn.Sequential(
                nn.Linear(in_features=self.gen_input, out_features=l),
                #nn.BatchNorm1d(l),
                nn.ReLU(),
                nn.Dropout(self.p)
                )
            )
            self.gen_input = l
            
        gen.append(nn.Sequential(
            nn.Linear(in_features=self.gen_input, out_features=self.img_shape)
            #,
            #nn.Sigmoid()
            )
        )
                   
        self.generat = nn.Sequential(*gen)

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        return self.generat(x)
                   
# DISCRIMINATOR    
class discriminator_simple_dense(nn.Module):
    def __init__(self, disc_layers:list, img_shape:int, p):
        super().__init__()
        self.disc_layers = disc_layers
        self.disc_input = img_shape
        self.p = p

    
        disc = []
        for l in disc_layers:
            disc.append(nn.Sequential(
                nn.Linear(in_features=self.disc_input, out_features=l),
                nn.BatchNorm1d(l),
                nn.ReLU(),
                nn.Dropout(self.p)
                )
            )
            self.disc_input = l
        # Add logits output
        disc.append(nn.Sequential(
            nn.Linear(in_features=self.disc_input, out_features=1)
            ,nn.Sigmoid()
            )
        )
        self.discrim = nn.Sequential(*disc)
        
    def forward(self, x):
        return self.discrim(x)

class discriminator_simple_dense_cls(nn.Module):
    def __init__(self, disc_layers:list, img_shape:int, p, n_classes):
        super().__init__()
        self.disc_layers = disc_layers
        self.disc_input = img_shape
        self.p = p

    
        disc = []
        for l in disc_layers:
            disc.append(nn.Sequential(
                nn.Linear(in_features=self.disc_input, out_features=l),
                nn.BatchNorm1d(l),
                nn.ReLU(),
                nn.Dropout(self.p)
                )
            )
            self.disc_input = l
        # Add logits output
        disc.append(nn.Sequential(
            nn.Linear(in_features=self.disc_input, out_features=1)
            ,nn.Sigmoid()
            )
        )
        self.discrim = nn.Sequential(*disc)
        self.aux_classifier = nn.Sequential(
            nn.Linear(in_features=img_shape, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.discrim(x), self.aux_classifier(x)

class c_discriminator_simple_dense(nn.Module):
    def __init__(self, disc_layers:list, img_shape:int, p):
        super().__init__()
        self.disc_layers = disc_layers
        self.disc_input = img_shape + 1
        self.p = p

    
        disc = []
        for l in disc_layers:
            disc.append(nn.Sequential(
                nn.Linear(in_features=self.disc_input, out_features=l),
                #nn.BatchNorm1d(l),
                nn.ReLU(),
                nn.Dropout(self.p)
                )
            )
            self.disc_input = l
        # Add logits output
        disc.append(nn.Sequential(
            nn.Linear(in_features=self.disc_input, out_features=1)
            #,
            #nn.Sigmoid()
            )
        )
        self.discrim = nn.Sequential(*disc)
        
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        return self.discrim(x)
# ## Conditional VAE

class conditional_simple_vae(nn.Module):
    """
    https://github.com/unnir/cVAE/blob/master/cvae.py
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
    Differnce between conditional and concatination?
    One-hot-encoding vs nn.Embedding
    https://spltech.co.uk/in-pytorch-what-is-nn-embedding-for-and-how-is-it-different-from-one-hot-encding-for-representing-categorical-data/?utm_content=cmp-true
    HERE: Condition on Semnatic space, not label!
    """
    def __init__(self, hidden_layers, input_dim, latent_dim, conditional_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = len(hidden_layers)
        self.conditional_dim = conditional_dim
        
        enc = []
        # Conditional layer
        enc.append(nn.Sequential(nn.Linear(in_features=self.input_dim+self.conditional_dim, out_features=self.input_dim)))
        # Encoding layer
        for l in hidden_layers:
            enc.append(
                nn.Sequential(
                    nn.Linear(in_features=self.input_dim, out_features=l),
                    nn.BatchNorm1d(l),
                    nn.ReLU()
                )
            )
            self.input_dim = l
        self.encode = nn.Sequential(*enc)
        self.encode_mu = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        self.encode_var = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        
        dec = []
        # Conditional layer
        dec.append(nn.Sequential(nn.Linear(in_features=self.latent_dim+self.conditional_dim, out_features=hidden_layers[-1])))
        # Decoding layer
        for i in range(1,self.num_layers):
            dec.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[self.num_layers-i], out_features=hidden_layers[self.num_layers-i-1]),
                    nn.BatchNorm1d(hidden_layers[self.num_layers-i-1]),
                    nn.ReLU()
                )
            )
        #Add sigmoid output last layer
        dec.append(nn.Sequential(
            nn.Linear(in_features=hidden_layers[0], out_features=self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Sigmoid()
            )
        )
        self.decode=nn.Sequential(*dec)
        
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, data, conditional_data):
        inputs = torch.cat([data, conditional_data], 1)
        z = self.encode(inputs)
        z_mu, z_var = self.encode_mu(z), self.encode_var(z)
        return self.reparameterize(z_mu, z_var), z_mu, z_var
    
    def decoder(self, data, conditional_data):
        inputs = torch.cat([data, conditional_data], 1)
        return self.decode(inputs)
    
    def forward(self, x, cond_x):
        # TODO: add reparameterization trick
        z, z_mu, z_var = self.encoder(x, cond_x)
        return self.decoder(z, cond_x), z_mu, z_var
                           

class cvae_simple_encoder(nn.Module):
    def __init__(self, hidden_layers, input_dim, latent_dim, conditional_dim, p):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = len(hidden_layers)
        self.conditional_dim = conditional_dim
        self.p = p
        
        enc = []
        # Conditional layer
        enc.append(nn.Sequential(nn.Linear(in_features=self.input_dim+self.conditional_dim, out_features=self.input_dim)))
        # Encoding layer
        for l in hidden_layers:
            enc.append(
                nn.Sequential(
                    nn.Linear(in_features=self.input_dim, out_features=l),
                    nn.BatchNorm1d(l),
                    nn.ReLU(),
                    nn.Dropout(self.p)
                )
            )
            self.input_dim = l
        self.encode = nn.Sequential(*enc)
        self.encode_mu = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        self.encode_var = nn.Linear(in_features=hidden_layers[-1], out_features=self.latent_dim)
        
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, conditional_data):
        self.inputs_enc = torch.cat([x, conditional_data], 1)
        z = self.encode(self.inputs_enc)
        z_mu, z_std = self.encode_mu(z), self.encode_var(z)
        return self.reparameterize(z_mu, z_std), z_mu, z_std


class cvae_simple_decoder(nn.Module):
    def __init__(self, hidden_layers, input_dim, latent_dim, conditional_dim, p):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = len(hidden_layers)
        self.conditional_dim = conditional_dim
        self.p = p
        
        dec = []
        # Conditional layer
        dec.append(nn.Sequential(nn.Linear(in_features=self.latent_dim+self.conditional_dim, out_features=hidden_layers[-1])))
        # Decoding layer
        for i in range(1,self.num_layers):
            dec.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_layers[self.num_layers-i], out_features=hidden_layers[self.num_layers-i-1]),
                    nn.BatchNorm1d(hidden_layers[self.num_layers-i-1]),
                    nn.ReLU(),
                    nn.Dropout(self.p)
                )
            )
        #Add sigmoid output last layer
        dec.append(nn.Sequential(
            nn.Linear(in_features=hidden_layers[0], out_features=self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.Sigmoid()
            )
        )
        self.decode=nn.Sequential(*dec)
    
    def forward(self, x, conditional_data):
        self.inputs_dec = torch.cat([x, conditional_data], 1)
        return self.decode(self.inputs_dec)


class cvae_simple(nn.Module):
    def __init__(self, hidden_layers, input_dim, latent_dim, conditional_dim, p):
        super().__init__()
        self.encoder= cvae_simple_encoder(hidden_layers, input_dim, latent_dim, conditional_dim, p)
        self.decoder= cvae_simple_decoder(hidden_layers, input_dim, latent_dim, conditional_dim, p)
        
    def forward(self, x, conditional_data):
        z, zm, zs = self.encoder(x, conditional_data)
        return self.decoder(z, conditional_data), zm, zs
    
    def loss_function(self, recon_x, x, mu, logvar, kl_weight):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_KLD = KLD.sum()
        return BCE + (kl_weight*total_KLD), BCE, total_KLD, KLD



# ## Data Loader

class custom_dataset_lbl(torch.utils.data.Dataset):
    def __init__(self, X_train, S_train, Y_train, transform=None):
        super().__init__()
        self.Xtr = X_train
        self.Str = S_train
        self.Ytr = Y_train
        self.trans = transform
    
    def __len__(self):
        return len(self.Xtr)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #x = torch.from_numpy(self.Xtr[idx]).view(1,1,-1).float()
        x = torch.from_numpy(self.Xtr[idx]).float()
        s = torch.from_numpy(self.Str[idx]).float()
        #y = torch.from_numpy(self.Ytr[idx]).int()
        y = self.Ytr[idx]
        
        if self.trans:
            x = self.trans(x)
        
        #x = torch.reshape(x, (x.shape[0],32,64))
        #y = torch.from_numpy(self.Ytr[idx]).view(1,1,-1).float()
        return x,s,y


"""
class custom_dataset_split(torch.utils.Dataset):
    """"""
        Return random splitted training and test set.
        Input:
            X: concatenated[X_tr,X_te]
            S: concatenated[S_tr,S_te]
            Y: concatenated[Y_tr,Y_te]
    """"""
    def __init__(self, X, S, Y, split:float, transform=None):
        self.X = X
        self.S = S
        self.Y = Y
        self.split = split
        
    def __len__(self):
        return int(len(self.Y)*(1-self.split)
    
    def get_split(self):
        self.rand_indx = random.choices(range(0,len(Y)), k=int(len(self.Y)*(1-self.split)))
        #TODO: Index of test data
        #      Overlap ratio of train and tes
        #.     Return train and test data, or a seperate class for test data?
        #.     Easier to shuffle before dataset generator?
                   
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
"""        


class custom_dataset_cub(torch.utils.data.Dataset):
    def __init__(self, root_path, transformer=None):
        """
            Path to labels: root_path + '/classes.txt'
            train/test: root_path + 'train_test_split.txt'
            rooth_path = '/Users/will/data/CUB_200_2011/CUB_200_2011/'
        """
        self.att, self.att_cls = get_att(root_path + 'attributes/class_attribute_labels_continuous.txt',
                                         root_path + 'classes.txt')
        self.img = get_img(rooth_path + 'images.txt')
        self.lbl = get_lbl(rooth_path + 'image_class_labels.txt')
        self.seen_cls, self.unseen_cls = get_split(root_path + 'trainvalclasses.txt',
                                                   root_path + 'testclasses.txt')
        
        self.img_lst = []
        self.att_lst = []
        self.lbl_lst = []
        for i in range(len(self.img)):
            if self.img[i].split('/')[0] in self.seen_cls:
                self.img_lst.append(self.img[i])
                self.att_lst.append(self.att[self.lbl[i]-1])
                self.lbl_lst.append(self.att_cls[self.lbl[i]-1])
    def __len__(self):
        return len(self.img_lst)
    
    def __getitem__(self, index):
        X = self.img_lst[index]
        Y = self.lbl_lst[index]
        S = self.att_lst[index]
        return X,Y,S
        
    def get_split(self, path_train, path_test, proposed=True):
        if proposed:
            seen_cls_file = open(path_train)
            seen_cls = seen_cls_file.read().split('\n')[:-1]
            seen_cls = [x.split('.',1)[1:][0] for x in seen_cls]

            unseen_cls_file = open(path_test)
            unseen_cls = unseen_cls_file.read().split('\n')[:-1]
            unseen_cls = [x.split('.',1)[1:][0] for x in unseen_cls]
            
            return seen_cls, unseen_cls
        else:
            print('Only proposed splitt is implemented')
    
    def get_att(self, path_att_file, path_att_cls):
        att_file = open(path_att_file)
        att_cls_file = open(path_att_cls)
        att = att_file.read().split('\n')[:-1]
        att_cls = att_cls_file.read().split('\n')[:-1]
        return att, [x.split('.',1)[-1:][0] for x in att_cls]

    def get_lbl(self, path_lbl):
        lbl_file = open(path_lbl)
        lbl = lbl_file.read().split('\n')[:-1]
        return [int(x.split(' ')[-1:][0]) for x in lbl]
    
    def get_img(self, path_img):
        img_file = open(path_img)
        img = img_file.read().split('\n')[:-1]
        return [x.split('.', 1)[1:][0] for x in img]


# +
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
        
class ConditionalModel(nn.Module):
    def __init__(self, n_steps, dim_input:int = 2048):
        super(ConditionalModel, self).__init__()
        self.dim_input = dim_input
        self.lin1 = ConditionalLinear(self.dim_input, self.dim_input*2, n_steps)
        self.lin2 = ConditionalLinear(self.dim_input*2, self.dim_input*2, n_steps)
        self.lin3 = ConditionalLinear(self.dim_input*2, self.dim_input*2, n_steps)
        self.lin4 = nn.Linear(self.dim_input*2, self.dim_input)
    
    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)


# -

class EMA(object):
    def __init__(self, device, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


