import torch
import torchvision.transforms as transforms
from argparse import Namespace
# from utils.utils_aug import CutOut, Create_Albumentations_From_Name,RandomErasing

class Config:
    def __init__(self, scheduler_type):
        self.random_seed = 0
        self.plot_train_batch_count = 5
        self.custom_augment = transforms.Compose([])

        if scheduler_type == 'CosineAnnealingLR':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
            self.lr_scheduler_params = {
                'T_max': 100,  #30
                'eta_min': 1e-6 #1e-6
            }
        elif scheduler_type == 'ExponentialLR':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
            self.lr_scheduler_params = {
                'gamma': 0.95
            }
        elif scheduler_type == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR
            self.lr_scheduler_params = {
                'gamma': 0.9,#0.8
                'step_size': 2
            }
        else:
            raise ValueError('Invalid scheduler type')

    def _get_opt(self):
        config_dict = {name:getattr(self, name) for name in dir(self) if name[0] != '_'}
        return Namespace(**config_dict)

if __name__ == '__main__':
    config = Config('StepLR')
    print(config._get_opt())