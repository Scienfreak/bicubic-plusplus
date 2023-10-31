import os
import torch
import pytorch_lightning as pl

from utils import conf_utils
from utils import my_utils
from training import trainer

conf = conf_utils.get_config(path='configs/conf.yaml')

if __name__ == '__main__':
    pl_module = trainer.SRTrainer(conf)

    pretrained_path = conf['pretrained_path']
    ext = os.path.splitext(pretrained_path)[1]
    if ext == '.pth':
        pl_module.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
    elif ext == '.ckpt':
        pl_module.load_from_checkpoint(pretrained_path, conf, strict=conf['strict_load'])

    # Save the model weights to a text file (replace 'weights.txt' with your desired file name)
    my_utils.save_weights_to_txt(pl_module, 'weights.txt')

    # Iterate through the model's parameters and check their data types
    for name, param in pl_module.named_parameters():
        print(f'Layer: {name}, Data Type: {param.dtype}')
