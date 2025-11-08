import sys
sys.path.append('..')
import importlib
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import code_package.model_package as mcr2_model
import code_package.mcr2_trainer as mcr2_trainer
from code_package import run_code_timer
timer =  run_code_timer.Timer()
import os
import warnings
warnings.filterwarnings("ignore")
import importlib
importlib.reload(mcr2_model)
importlib.reload(mcr2_trainer)
import yaml

date = '20251103'
new_path = 'D:\linan\code_simulated_data\LRD_project\code_mnist'
sys.path.append(new_path)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
digit_list =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
timer.start()
for i in [2,3,4,5,6,7,8,9,10]:
    ratio = 0
    num_list = digit_list[:i]
    latent_dim = len(num_list)
    class_dim = len(num_list)

    # 读取数据
    data_dir =os.path.join('./dataloaders/', 'latent_dim_'+str(latent_dim), 'label_ratio_{ratio}'.format(ratio=ratio))
    data_loader, data_set = torch.load(os.path.join(data_dir, 'dataset.pth'), map_location=DEVICE)
     

    # 准备模型
    params_path = os.path.join('model_params', 'mnist_ae.yaml')
    mnist_params = yaml.load(open(params_path, 'r'), Loader=yaml.FullLoader)

    encoder_params = mnist_params['encoder_params']
    decoder_params = mnist_params['decoder_params']
    #kl_weights = mnist_params['kl_weights']
    encoder_params['class_dim'] = class_dim
    encoder_params['latent_dim'] = latent_dim
    decoder_params['class_dim'] = class_dim
    decoder_params['latent_dim'] = latent_dim

    for kl_weights in [1e-5]:
       
        mcr2 =  mcr2_model.CVAE(encoder_params, decoder_params, is_ts=0, kl_weights=kl_weights)
        
        mcr2.to(DEVICE)
        
        config_path = os.path.join('configs', 'mnist_cvae_config.yaml')
        
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        config['kl_weights'] = kl_weights
        config['n_epoch'] = 500
        
        base_dir =os.path.join(new_path, f'test_al_{date}_vae', 'latent_dim_'+str(latent_dim), 'label_ratio_' + str(ratio),'kl_' + str(kl_weights))
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        trainer  = mcr2_trainer.MCR2Trainer(mcr2, data_loader, data_set, config, base_dir, DEVICE)
        best_overall_model_path, loss = trainer.multi_start_train([1])
        torch.save(loss, os.path.join(base_dir, 'val_loss.pth'))

timer.stop(is_print=1)