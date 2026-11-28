"""
UnetFormer for uavidpp datasets with supervision training
Libo Wang, 2022.02.22
Adapted: David Rafael Rotar, 2026.11.26
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavidpp_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 40
ignore_index = 255
train_batch_size = 4
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "unetformer-swinv2-small-1024-e40"
weights_path = "model_weights/uavidpp/{}".format(weights_name)
test_weights_name = weights_name # "last"
log_name = 'uavidpp/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = [0]  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = UNetFormer(
    backbone_name='swinv2_small_window8_256',
    num_classes=num_classes,
    img_size=1024,
    grad_checkpointing=True
)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

# define the dataloader

train_dataset = UAVIDPPDataset(data_root='data/uavidpp/train', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

val_dataset = UAVIDPPDataset(data_root='data/uavidpp/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

