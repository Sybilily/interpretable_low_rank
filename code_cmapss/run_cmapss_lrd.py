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
params_path = os.path.join('model_params','cmapss_lrd.yaml')
cmapss_lrd_params = yaml.load(open(params_path, 'r'), Loader=yaml.FullLoader)
configs = yaml.load(open('configs/cmapss_mcr2_config.yaml', 'r'), Loader=yaml.FullLoader)

encoder_params = cmapss_lrd_params['encoder_params']
decoder_params = cmapss_lrd_params['decoder_params']

decoder_params['decoder_depth'] = 2
decoder_params['decoder_width'] = [200,50]
decoder_params['recons_activation'] = 'None'

encoder_params['encoder_activation'] = 'leaky_relu'
decoder_params['decoder_activation'] = 'leaky_relu'
encoder_params['cov_activation'] = 'sigmoid'

DEVICE  = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

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
    decoder_params['drop_rate'] = 0.1
    lambdas = 1e-20
    mcr2 =  mcr2_model.MCR2(encoder_params, decoder_params, is_ts=0, kl_weights=kl_weights)
    mcr2.lambdas = lambdas
    mcr2.to(DEVICE)
    mcr2.device = DEVICE
    base_dir =os.path.join('./res_lrd/', 'kl_'+str(kl))
    configs['pre_train_epoch'] = 500
    configs['n_epoch'] = 100
    configs['save_freq'] = 150
    configs['kl_weights'] = kl_weights
    configs['pre_train_path'] = 'pre_train.pth'
    configs['weight_ls'] = [1,20,1,0]
    trainer  = mcr2_trainer.MCR2Trainer(mcr2, data_loader, data_set, configs, base_dir, DEVICE)
    best_overall_model_path, val_loss_ls = trainer.multi_start_train(list(range(1,11)))
    print(val_loss_ls)
    timer.stop(is_print=True)