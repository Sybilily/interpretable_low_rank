import os
import torch
import importlib
import data_pre
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")
import code_package.model_package as mcr2_model
import code_package.mcr2_trainer as mcr2_trainer
from code_package import run_code_timer
timer =  run_code_timer.Timer()

import yaml
params_path = os.path.join('model_params','cmapss_ae.yaml')
cmapss_lrd_params = yaml.load(open(params_path, 'r'), Loader=yaml.FullLoader)
configs = yaml.load(open('configs/cmapss_cvae_config.yaml', 'r'), Loader=yaml.FullLoader)

encoder_params = cmapss_lrd_params['encoder_params']
decoder_params = cmapss_lrd_params['decoder_params']
decoder_params['recons_activation'] = None

DEVICE  = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

for kl in [1e-3]:
    kl_weights = kl#cmapss_lrd_params['kl_weights']
    latent_dim = 3
    datadir = os.path.join('dataloaders',str(latent_dim), 'cmapss_004.pt')
    data_loader, data_set = torch.load(datadir, map_location=DEVICE)
    timer.start()
    class_dim = latent_dim
    encoder_params['class_dim'] = class_dim
    encoder_params['latent_dim'] = latent_dim
    decoder_params['class_dim'] = class_dim
    decoder_params['latent_dim'] = latent_dim
    lambdas = 1e-13
    mcr2 =  mcr2_model.CVAE(encoder_params, decoder_params, is_ts=0, kl_weights=kl_weights)
    mcr2.lambdas = lambdas
    mcr2.to(DEVICE)
    mcr2.device = DEVICE
    base_dir =os.path.join('./res_cvae/', 'kl_'+str(kl))
    configs['pre_train_epoch'] = 300
    configs['n_epoch'] = 600
    configs['save_freq'] = 300
    configs['kl_weights'] = kl_weights
    trainer  = mcr2_trainer.MCR2Trainer(mcr2, data_loader, data_set, configs, base_dir, DEVICE)
    best_overall_model_path, loss = trainer.multi_start_train(list(range(1, 11)))
    torch.save(loss, os.path.join(base_dir, 'val_loss_ls.pt'))
    timer.stop(is_print=True)