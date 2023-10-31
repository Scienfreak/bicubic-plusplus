import os
import torch
import pytorch_lightning as pl

from utils import conf_utils
from utils import my_utils
from training import trainer
conf = conf_utils.get_config(path='configs/conf.yaml')

from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
tuning_criterion = TuningCriterion(max_trials=600)
conf_ptq = PostTrainingQuantConfig(
    approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
)

from neural_compressor.quantization import fit




if __name__ == '__main__':
    pl_module = trainer.SRTrainer(conf)

    pretrained_path = conf['pretrained_path']
    ext = os.path.splitext(pretrained_path)[1]
    if ext == '.pth':
        pl_module.network.load_state_dict(torch.load(pretrained_path), strict=conf['strict_load'])
    elif ext == '.ckpt':
        pl_module.load_from_checkpoint(pretrained_path, conf, strict=conf['strict_load'])

    pl_module.eval()

    trainer = pl.Trainer(gpus=[])

    ###
    def eval_func_for_nc(model_n, trainer_n):
        setattr(pl_module, "model", model_n)
        result = trainer_n.validate(model=pl_module, dataloaders=pl_module.val_dataloader())
        return result[0]["accuracy"]


    def eval_func(model):   
        return eval_func_for_nc(model, trainer)


    model_quantized = fit(model=pl_module, conf=conf_ptq, calib_dataloader=pl_module.val_dataloader(), eval_func=eval_func)

    result = trainer.validate(model_quantized)
