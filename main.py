import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, argparse, shutil, random, imp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch, torchvision, time, datetime, copy
from sklearn.utils.class_weight import compute_class_weight
from copy import deepcopy
from utils.utils_fit import fitting, fitting_distill, EarlyStopping
from utils.utils_model import select_model
from utils import utils_aug
from utils.utils import save_model, plot_train_batch, WarmUpLR, show_config, setting_optimizer, check_batch_size, \
    plot_log, update_opt, load_weights, get_channels, dict_to_PrettyTable, ModelEMA, select_device
from utils.utils_distill import *
from utils.utils_loss import *
import pandas as pd
from mainweb_globalValue import stop_training_flag
from torch.utils.tensorboard import SummaryWriter
import wandb


'''启用cudnn的自动优化（只有卷积不变才可以，动态卷积不能用）。当输入数据的大小不变时，这可以让内置的cudnn自动寻找最适合当前配置的高效算法，来达到优化运行效率的目的。
但是这样可能会使得每次运行的结果不一样，因为cudnn会自动选择最适合的算法。
如果需要固定结果，可以将torch.backends.cudnn.benchmark设置为False'''
# torch.backends.cudnn.benchmark=True
torch.backends.cudnn.benchmark=False
# 下面这行代码也是固定随机种子，确保对于相同的输入，得到相同的输出
# 设置PyTorch的cuDNN后端为确定性模式。这将确保每次运行程序时，对于相同的输入，卷积等操作的输出是确定的。这是通过禁用cuDNN的某些非确定性的算法来实现的。
torch.backends.cudnn.deterministic = True
def set_seed(seed):
    # 在需要重现结果的时候固定随机种子，这样每次运行的结果都是一样的
    # 自己造模型的时候不用固定！
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_options(opt, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            setattr(opt, key, value)

def parse_opt(model_name_in, pretrained_in, batch_size_in, epoch_in, save_path_in, loss_in, optimizer_in, scheduler_lr_in,lr_in, amp_in, warm_up_in, warmup_ratios_in, metrice_in, patience_in):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='repconvnest', help='model name')
    parser.add_argument('--pretrained', action="store_true", help='using pretrain weight')
    parser.add_argument('--weight', type=str, default='', help='loading weight path')
    parser.add_argument('--config', type=str, default='config/config.py', help='config path')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    parser.add_argument('--image_channel', type=int, default=3, help='image channel')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (-1 for autobatch)')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--resume', action="store_true", help='resume from save_path traning')

    # optimizer parameters
    parser.add_argument('--loss', type=str, choices=['PolyLoss', 'CrossEntropyLoss', 'FocalLoss','FocalMarginLoss','RDropLoss','PSoftmaxLoss','CapsuleLoss'],
                        default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--scheduler_lr', type=str, choices=['CosineAnnealingLR', 'ExponentialLR', 'StepLR'],
                        default='CosineAnnealingLR', help='lr_scheduler')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW', 'RMSProp'], default='AdamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--class_balance', action="store_true", help='using class balance in loss')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum in optimizer')
    parser.add_argument('--amp', action="store_true", help='using AMP(Automatic Mixed Precision)')
    parser.add_argument('--warmup', action="store_true", help='using WarmUp LR')
    parser.add_argument('--warmup_ratios', type=float, default=0.01,
                        help='warmup_epochs = int(warmup_ratios * epoch) if warmup=True')
    parser.add_argument('--warmup_minlr', type=float, default=1e-6,
                        help='minimum lr in warmup(also as minimum lr in training)')
    parser.add_argument('--metrice', type=str, choices=['loss', 'acc', 'mean_acc'], default='mean_acc', help='best.pt save relu')
    parser.add_argument('--patience', type=int, default=20, help='EarlyStopping patience (--metrice without improvement)')

    # Data Processing parameters
    parser.add_argument('--imagenet_meanstd', action="store_true", help='using ImageNet Mean and Std')
    parser.add_argument('--mixup', type=str, choices=['mixup', 'cutmix', 'none'], default='none', help='MixUp Methods')
    parser.add_argument('--Augment', type=str,
                        choices=['RandAugment', 'AutoAugment', 'TrivialAugmentWide', 'AugMix', 'CutOut','RandomErasing','none'], default='none',
                        help='Data Augment')
    parser.add_argument('--test_tta', action="store_true", help='using TTA')

    # Knowledge Distillation parameters
    parser.add_argument('--kd', action="store_true", help='Knowledge Distillation')
    parser.add_argument('--kd_ratio', type=float, default=0.7, help='Knowledge Distillation Loss ratio')
    parser.add_argument('--kd_method', type=str, choices=['SoftTarget', 'MGD', 'SP', 'AT'], default='SoftTarget', help='Knowledge Distillation Method')
    parser.add_argument('--teacher_path', type=str, default='', help='teacher model path')

    # Tricks parameters
    parser.add_argument('--rdrop', action="store_true", help='using R-Drop')
    # parser.add_argument('--ema', action="store_true", help='using EMA(Exponential Moving Average) Reference to YOLOV5')
    parser.add_argument('--ema', default=True, type=bool, help='using EMA(Exponential Moving Average) Reference to YOLOV5')
    opt = parser.parse_known_args()[0]

    set_options(opt, model_name=model_name_in, pretrained=pretrained_in, batch_size=int(batch_size_in),
                epoch=int(epoch_in), save_path=save_path_in, loss=loss_in, optimizer=optimizer_in,
                scheduler_lr=scheduler_lr_in, lr=float(lr_in), amp=amp_in, warmup=warm_up_in, warmup_ratios=float(warmup_ratios_in),
                metrice=metrice_in, patience=int(patience_in))

    if opt.resume:
        opt.resume = True
        if not os.path.exists(os.path.join(opt.save_path, 'last.pt')):
            raise Exception('last.pt not found. please check your --save_path folder and --resume parameters')
        ckpt = torch.load(os.path.join(opt.save_path, 'last.pt'))
        opt = ckpt['opt']
        opt.resume = True
        print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    else:
        if os.path.exists(opt.save_path):
            shutil.rmtree(opt.save_path)
        os.makedirs(opt.save_path)
        config = imp.load_source('config', opt.config).Config(opt.scheduler_lr)
        shutil.copy(__file__, os.path.join(opt.save_path, 'main.py'))
        shutil.copy(opt.config, os.path.join(opt.save_path, 'config.py'))
        opt = update_opt(opt, config._get_opt())

    
    set_seed(opt.random_seed)
    show_config(deepcopy(opt))

    CLASS_NUM = len(os.listdir(opt.train_path))
    DEVICE = select_device(opt.device, opt.batch_size)

    train_transform, test_transform = utils_aug.get_dataprocessing(torchvision.datasets.ImageFolder(opt.train_path),
                                                                   opt)
    train_dataset = torchvision.datasets.ImageFolder(opt.train_path, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(opt.val_path, transform=test_transform)
    if opt.resume:
        # ckpt表示checkpoint检查点，用于保存模型的训练权重
        model = ckpt['model'].to(DEVICE).float()
    else:
        model = select_model(opt.model_name, CLASS_NUM, (opt.image_size, opt.image_size), opt.image_channel,
                             opt.pretrained)
        model = load_weights(model, opt).to(DEVICE)
        plot_train_batch(copy.deepcopy(train_dataset), opt,True) # 输出训练集的图片
        plot_train_batch(copy.deepcopy(test_dataset), opt,False) # 输出测试集的图片

    batch_size = opt.batch_size if opt.batch_size != -1 else check_batch_size(model, opt.image_size, amp=opt.amp)

    if opt.class_balance:
        class_weight = np.sqrt(compute_class_weight('balanced', classes=np.unique(train_dataset.targets), y=train_dataset.targets))
    else:
        class_weight = np.ones_like(np.unique(train_dataset.targets))
    print('class weight: {}'.format(class_weight))

    '''测试集和训练集都是随机打乱读取，sampler可以设置采样方法，但是这里没有设置'''
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=opt.workers,pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(test_dataset, max(batch_size // (10 if opt.test_tta else 1), 1),
                                               shuffle=True, num_workers=(0 if opt.test_tta else opt.workers),pin_memory=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(opt.amp if torch.cuda.is_available() else False))
    ema = ModelEMA(model) if opt.ema else None
    optimizer = setting_optimizer(opt, model)
    lr_scheduler = WarmUpLR(optimizer, opt)
    if opt.resume:
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        loss = ckpt['loss'].to(DEVICE)
        scaler.load_state_dict(ckpt['scaler'])
        if opt.ema:
            ema.ema = ckpt['ema'].to(DEVICE).float()
            ema.updates = ckpt['updates']
    else:
        # 此处weight用于计算损失,eval(opt.loss)返回的是一个类，这个类的实例化对象就是一个损失函数
        # eval用于将字符串转换为python表达式
        loss = eval(opt.loss)(label_smoothing=opt.label_smoothing,
                              weight=torch.from_numpy(class_weight).to(DEVICE).float())
        if opt.rdrop:
            loss = RDropLoss(loss)
    return opt, model, ema, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, (
        ckpt['epoch'] if opt.resume else 0), (ckpt['best_metrice'] if opt.resume else None)


# 下面为我自己写的函数


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    # 返回保留两位小数的模型大小
    return round(all_size,2)

def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    # 全部向下取整
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return hours, minutes, seconds

def convert_seconds_to_minutes(seconds):
    minutes = seconds / 60
    return round(minutes, 2)


# 用于gradio界面函数
def train_main(model_name_in, pretrained_in, batch_size_in, epoch_in, save_path_in, loss_in, optimizer_in, scheduler_lr_in,lr_in, amp_in, warm_up_in, warmup_ratios_in, metrice_in, patience_in):
    global stop_training_flag
    # 设置CUDA_LAUNCH_BLOCKING=1，可以让程序在报错时停止，方便调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # 从配置中解析出各种参数和对象，包括优化器、模型、数据集、学习率调度器、损失函数等
    opt, model, ema, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, begin_epoch, best_metrice = parse_opt(model_name_in, pretrained_in, batch_size_in, epoch_in, save_path_in, loss_in, optimizer_in, scheduler_lr_in, lr_in, amp_in, warm_up_in, warmup_ratios_in, metrice_in, patience_in)

    # 初始化wandb
    wandb.init(
        project="oil-well-diagnosis",
        name=opt.save_path.split('/')[-1] if '/' in opt.save_path else opt.save_path.split('\\')[-1],
        config={
            "model_name": opt.model_name,
            "batch_size": opt.batch_size,
            "epoch": opt.epoch,
            "learning_rate": opt.lr,
            "optimizer": opt.optimizer,
            "loss": opt.loss,
            "scheduler": opt.scheduler_lr,
            "amp": opt.amp,
            "warmup": opt.warmup,
            "pretrained": opt.pretrained
        }
    )

    # 如果不是从上次的训练结果恢复，那么就创建一个新的训练日志文件
    if not opt.resume:
        save_epoch = 0
        with open(os.path.join(opt.save_path, 'train.log'), 'w+') as f:
            if opt.kd:
                f.write('epoch,lr,loss,kd_loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
            else:
                f.write('epoch,lr,loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
    else:
        # 如果是从上次的训练结果恢复，那么就加载上次的训练进度
        save_epoch = torch.load(os.path.join(opt.save_path, 'last.pt'))['best_epoch']

    # 如果开启了知识蒸馏
    if opt.kd:
        # 检查教师模型是否存在
        if not os.path.exists(os.path.join(opt.teacher_path, 'best.pt')):
            raise Exception('teacher best.pt not found. please check your --teacher_path folder')
        # 加载教师模型
        teacher_ckpt = torch.load(os.path.join(opt.teacher_path, 'best.pt'))
        teacher_model = teacher_ckpt['model'].float().to(DEVICE).eval()
        print('found teacher checkpoint from {}, model type:{}\n{}'.format(opt.teacher_path, teacher_model.name, dict_to_PrettyTable(teacher_ckpt['best_metrice'], 'Best Metrice')))

        # 如果是从上次的训练结果恢复，那么就加载上次的知识蒸馏损失
        if opt.resume:
            kd_loss = torch.load(os.path.join(opt.save_path, 'last.pt'))['kd_loss'].to(DEVICE)
        else:
            # 否则，根据知识蒸馏的方法来初始化知识蒸馏损失
            if opt.kd_method == 'SoftTarget':
                kd_loss = SoftTarget().to(DEVICE)
            elif opt.kd_method == 'MGD':
                kd_loss = MGD(get_channels(model, opt), get_channels(teacher_model, opt)).to(DEVICE)
                optimizer.add_param_group({'params': kd_loss.parameters(), 'weight_decay': opt.weight_decay})
            elif opt.kd_method == 'SP':
                kd_loss = SP().to(DEVICE)
            elif opt.kd_method == 'AT':
                kd_loss = AT().to(DEVICE)

    # 在训练循环中使用早停
    early_stopping = EarlyStopping(patience=opt.patience, delta=1e-5)
    # 新建一个dataframe，用于记录训练过程中的指标
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss', 'test_loss','train_acc', 'test_acc', 'train_mean_acc', 'test_mean_acc'])
    # 在dataframe中添加一行初始化，所有数据都为0
    df = df._append({'epoch': 0, 'lr': 0, 'train_loss': 0, 'test_loss': 0, 'train_acc': 0, 'test_acc': 0, 'train_mean_acc': 0, 'test_mean_acc': 0}, ignore_index=True)
    # 在opt.save_path文件夹下创建一个名为'训练过程记录.txt'的文件
    with open(os.path.join(opt.save_path, '训练过程记录.txt'), 'w+') as f:
        f.write('训练过程记录\n')
        f.write('模型名称:{}\n'.format(opt.model_name))
        f.write('模型保存路径:{}\n'.format(opt.save_path))
        f.write('训练时间:{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        f.write('训练参数:\n')
        f.write('是否使用预训练权重:{}\n'.format(opt.pretrained))
        f.write('batch_size:{}\n'.format(opt.batch_size))
        f.write('epoch:{}\n'.format(opt.epoch))
        f.write('学习率:{}\n'.format(opt.lr))
        f.write('优化器:{}\n'.format(opt.optimizer))
        f.write('损失函数:{}\n'.format(opt.loss))
        f.write('是否使用AMP:{}\n'.format(opt.amp))
        f.write('是否使用WarmUp:{}\n'.format(opt.warmup))
        f.write('WarmUp轮数:{}\n'.format(opt.warmup_ratios))
        f.write('最优指标:{}\n'.format(opt.metrice))
        f.write('早停轮数:{}\n'.format(opt.patience))

    time_begin_mian = time.perf_counter()
    """开始训练"""
    for epoch in range(begin_epoch, opt.epoch):
        begin = time.perf_counter()
        if stop_training_flag:
            with open(os.path.join(opt.save_path, '训练过程记录.txt'), 'w+') as f:
                print('训练中止！\n', file=f)
            yield df
        if early_stopping.early_stop:
            break
        # 如果开启了知识蒸馏，那么就使用知识蒸馏的训练方法
        if opt.kd:
            metrice = fitting_distill(teacher_model, model, ema, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, '{}/{}'.format(epoch + 1,opt.epoch), opt)
        else:
            # 否则，就使用普通的训练方法
            metrice = fitting(model, ema, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler,'{}/{}'.format(epoch + 1, opt.epoch), opt)
            # 如果训练中止，那么就返回训练过程记录.txt文件路径和dataframe
            if stop_training_flag:
                with open(os.path.join(opt.save_path, '训练过程记录.txt'), 'w+') as f:
                    print('训练中止！\n', file=f)
                yield df
        # 将训练的结果写入到训练日志中
        with open(os.path.join(opt.save_path, 'train.log'), 'a+') as f:
            f.write(
                '\n{},{:.10f},{}'.format(epoch + 1, optimizer.param_groups[2]['lr'], metrice[1]))

        # 获取当前的学习率
        n_lr = optimizer.param_groups[2]['lr']
        # 更新学习率
        lr_scheduler.step()

        # 如果是第一次训练，那么就初始化最好的指标
        if best_metrice is None:
            best_metrice = metrice[0]
            save_model(
                    os.path.join(opt.save_path, 'best.pt'),
                    **{
                    'model': (deepcopy(ema.ema).to('cpu').half() if opt.ema else deepcopy(model).to('cpu').half()),
                    'opt': opt,
                    'best_metrice': best_metrice,
                    }
                )
            save_epoch = epoch
        else:
            # 如果当前的指标比最好的指标还要好，那么就更新最好的指标，并保存模型
            if eval('{} {} {}'.format(metrice[0]['test_{}'.format(opt.metrice)], '<' if opt.metrice == 'loss' else '>', best_metrice['test_{}'.format(opt.metrice)])):
                best_metrice = metrice[0]
                save_model(
                    os.path.join(opt.save_path, 'best.pt'),
                    **{
                    'model': (deepcopy(ema.ema).to('cpu').half() if opt.ema else deepcopy(model).to('cpu').half()),
                    'opt': opt,
                    'best_metrice': best_metrice,
                    }
                )
                save_epoch = epoch

        # 保存当前的训练状态
        save_model(
            os.path.join(opt.save_path, 'last.pt'),
            **{
               'model': deepcopy(model).to('cpu').half(),
               'ema': (deepcopy(ema.ema).to('cpu').half() if opt.ema else None),
               'updates': (ema.updates if opt.ema else None),
               'opt': opt,
               'epoch': epoch + 1,
               'optimizer' : optimizer.state_dict(),
               'lr_scheduler': lr_scheduler.state_dict(),
               'best_metrice': best_metrice,
               'loss': deepcopy(loss).to('cpu'),
               'kd_loss': (deepcopy(kd_loss).to('cpu') if opt.kd else None),
               'scaler': scaler.state_dict(),
               'best_epoch': save_epoch,
            }
        )

        # 将当前的训练状态写入训练过程记录.txt文件中
        with open(os.path.join(opt.save_path, '训练过程记录.txt'), 'a+') as f:
            print(dict_to_PrettyTable(metrice[0], '{} epoch:{}/{}, best_epoch:{}, time:{:.2f}s, lr:{:.8f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1, opt.epoch, save_epoch + 1, time.perf_counter() - begin, n_lr,
            )) + '\n', file=f)

        # 将当前的训练状态写入dataframe中，其中，将epoch添加进df['epoch']中，其他以此类推
        df = df._append({'epoch': epoch + 1, 'lr': n_lr,
                        'train_loss': metrice[0]['train_loss'], 'test_loss': metrice[0]['test_loss'],
                        'train_acc': metrice[0]['train_acc'], 'test_acc': metrice[0]['test_acc'],
                        'train_mean_acc': metrice[0]['train_mean_acc'], 'test_mean_acc': metrice[0]['test_mean_acc']}, ignore_index=True)

        # 记录到wandb
        wandb.log({
            'epoch': epoch + 1,
            'lr': n_lr,
            'train_loss': metrice[0]['train_loss'],
            'test_loss': metrice[0]['test_loss'],
            'train_acc': metrice[0]['train_acc'],
            'test_acc': metrice[0]['test_acc'],
            'train_mean_acc': metrice[0]['train_mean_acc'],
            'test_mean_acc': metrice[0]['test_mean_acc'],
        })

        # 如果在一定的训练周期内训练集损失没有继续下降，那么就提前停止训练
        early_stopping(metrice[0]['test_loss'], model)
        if early_stopping.early_stop or epoch == opt.epoch - 1:
            if early_stopping.early_stop:
                with open(os.path.join(opt.save_path, '训练过程记录.txt'), 'a+') as f:
                    print('No Improve for test_loss from {} to {}, EarlyStopping.'.format(save_epoch + 1, epoch)+'\n',file=f)
            # 模型训练完毕，计算训练时间,格式化字符串，表示时分秒
            time_end = time.perf_counter()
            # 绘制训练日志的图表
            plot_log(opt)
            # 计算best.pt模型内存
            best_model_path = os.path.join(opt.save_path, 'best.pt')
            best_model = torch.load(best_model_path)['model']
            # 获取模型大小(模型参数数量)
            best_model_size  = getModelSize(best_model)
            # 计算训练时间
            time_train = time_end - time_begin_mian
            hours, minutes, seconds = convert_seconds(time_train)
            with open(os.path.join(opt.save_path, '训练过程记录.txt'), 'a+') as f:
                # 添加一句训练结束
                print('训练结束！\n', file=f)
                print(f"训练时间为: {hours}小时{minutes}分钟{seconds}秒\n", file=f)
                print(f"最优模型的内存大小为 {best_model_size} Mb.\n", file=f)

        # 结束wandb
        wandb.finish()

        yield df


if __name__ == '__main__':
    # 设置CUDA_LAUNCH_BLOCKING=1，可以让程序在报错时停止，方便调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # 从配置中解析出各种参数和对象，包括优化器、模型、数据集、学习率调度器、损失函数等
    opt, model, ema, train_dataset, test_dataset, optimizer, scaler, lr_scheduler, loss, DEVICE, CLASS_NUM, begin_epoch, best_metrice = parse_opt()

    # 初始化wandb
    wandb.init(
        project="oil-well-diagnosis",
        name=opt.save_path.split('/')[-1] if '/' in opt.save_path else opt.save_path.split('\\')[-1],
        config={
            "model_name": opt.model_name,
            "batch_size": opt.batch_size,
            "epoch": opt.epoch,
            "learning_rate": opt.lr,
            "optimizer": opt.optimizer,
            "loss": opt.loss,
            "scheduler": opt.scheduler_lr,
            "amp": opt.amp,
            "warmup": opt.warmup,
            "pretrained": opt.pretrained
        }
    )

    # 如果不是从上次的训练结果恢复，那么就创建一个新的训练日志文件
    if not opt.resume:
        save_epoch = 0
        with open(os.path.join(opt.save_path, 'train.log'), 'w+') as f:
            if opt.kd:
                f.write('epoch,lr,loss,kd_loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
            else:
                f.write('epoch,lr,loss,acc,mean_acc,test_loss,test_acc,test_mean_acc')
    else:
        # 如果是从上次的训练结果恢复，那么就加载上次的训练进度
        save_epoch = torch.load(os.path.join(opt.save_path, 'last.pt'))['best_epoch']

    # 如果开启了知识蒸馏
    if opt.kd:
        # 检查教师模型是否存在
        if not os.path.exists(os.path.join(opt.teacher_path, 'best.pt')):
            raise Exception('teacher best.pt not found. please check your --teacher_path folder')
        # 加载教师模型
        teacher_ckpt = torch.load(os.path.join(opt.teacher_path, 'best.pt'))
        teacher_model = teacher_ckpt['model'].float().to(DEVICE).eval()
        print('found teacher checkpoint from {}, model type:{}\n{}'.format(opt.teacher_path, teacher_model.name, dict_to_PrettyTable(teacher_ckpt['best_metrice'], 'Best Metrice')))

        # 如果是从上次的训练结果恢复，那么就加载上次的知识蒸馏损失
        if opt.resume:
            kd_loss = torch.load(os.path.join(opt.save_path, 'last.pt'))['kd_loss'].to(DEVICE)
        else:
            # 否则，根据知识蒸馏的方法来初始化知识蒸馏损失
            if opt.kd_method == 'SoftTarget':
                kd_loss = SoftTarget().to(DEVICE)
            elif opt.kd_method == 'MGD':
                kd_loss = MGD(get_channels(model, opt), get_channels(teacher_model, opt)).to(DEVICE)
                optimizer.add_param_group({'params': kd_loss.parameters(), 'weight_decay': opt.weight_decay})
            elif opt.kd_method == 'SP':
                kd_loss = SP().to(DEVICE)
            elif opt.kd_method == 'AT':
                kd_loss = AT().to(DEVICE)

 

    # 在训练循环中使用早停
    early_stopping = EarlyStopping(patience=opt.patience, delta=1e-5)


    # 打印开始训练的时间
    print('{} begin train!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    starttime = time.strftime("%Y-%m-%d_%H_%M_%S")
    writer = SummaryWriter(log_dir=os.path.join("log", starttime[:13]+'-'+str(opt.model_name)),comment=starttime[:13]+'-'+str(opt.model_name),flush_secs=60)
    # 记录开始训练的时间
    time_begin = time.perf_counter()
    """开始训练"""
    for epoch in range(begin_epoch, opt.epoch):

        begin = time.perf_counter()
        # 如果开启了知识蒸馏，那么就使用知识蒸馏的训练方法
        if opt.kd:
            metrice = fitting_distill(teacher_model, model, ema, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, '{}/{}'.format(epoch + 1,opt.epoch), opt)
        else:
            # 否则，就使用普通的训练方法
            metrice = fitting(model, ema, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler,'{}/{}'.format(epoch + 1, opt.epoch), opt)

        # 将训练的结果写入到训练日志中
        with open(os.path.join(opt.save_path, 'train.log'), 'a+') as f:
            f.write(
                '\n{},{:.10f},{}'.format(epoch + 1, optimizer.param_groups[2]['lr'], metrice[1]))

        # 获取当前的学习率
        n_lr = optimizer.param_groups[2]['lr']
        # 更新学习率
        lr_scheduler.step()
        # 更新学习率，以验证集的损失为指标
        # lr_scheduler.step(metrice[0]['test_loss'])

        # 如果是第一次训练，那么就初始化最好的指标
        if best_metrice is None:
            best_metrice = metrice[0]
        else:
            # 如果当前的指标比最好的指标还要好，那么就更新最好的指标，并保存模型
            if eval('{} {} {}'.format(metrice[0]['test_{}'.format(opt.metrice)], '<' if opt.metrice == 'loss' else '>', best_metrice['test_{}'.format(opt.metrice)])):
                best_metrice = metrice[0]
                save_model(
                    os.path.join(opt.save_path, 'best.pt'),
                    **{
                    'model': (deepcopy(ema.ema).to('cpu').half() if opt.ema else deepcopy(model).to('cpu').half()),
                    'opt': opt,
                    'best_metrice': best_metrice,
                    }
                )
                save_epoch = epoch

        # 保存当前的训练状态
        save_model(
            os.path.join(opt.save_path, 'last.pt'),
            **{
               'model': deepcopy(model).to('cpu').half(),
               'ema': (deepcopy(ema.ema).to('cpu').half() if opt.ema else None),
               'updates': (ema.updates if opt.ema else None),
               'opt': opt,
               'epoch': epoch + 1,
               'optimizer' : optimizer.state_dict(),
               'lr_scheduler': lr_scheduler.state_dict(),
               'best_metrice': best_metrice,
               'loss': deepcopy(loss).to('cpu'),
               'kd_loss': (deepcopy(kd_loss).to('cpu') if opt.kd else None),
               'scaler': scaler.state_dict(),
               'best_epoch': save_epoch,
            }
        )

        # 打印当前的训练状态
        print(dict_to_PrettyTable(metrice[0], '{} epoch:{}/{}, best_epoch:{}, time:{:.2f}s, lr:{:.8f}'.format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    epoch + 1, opt.epoch, save_epoch + 1, time.perf_counter() - begin, n_lr,
                )))

        writer.add_scalar('loss/train_loss', metrice[0]['train_loss'], epoch)
        writer.add_scalar('acc/train_acc', metrice[0]['train_acc'], epoch)
        writer.add_scalar('acc/train_mean_acc', metrice[0]['train_mean_acc'], epoch)
        writer.add_scalar('loss/val_loss',metrice[0]['test_loss'], epoch)
        writer.add_scalar('acc/val_acc',metrice[0]['test_acc'], epoch)
        writer.add_scalar('acc/val_mean_acc', metrice[0]['test_mean_acc'], epoch)
        writer.add_scalar('lr',n_lr, epoch)

        # 记录到wandb
        wandb.log({
            'epoch': epoch + 1,
            'lr': n_lr,
            'train_loss': metrice[0]['train_loss'],
            'test_loss': metrice[0]['test_loss'],
            'train_acc': metrice[0]['train_acc'],
            'test_acc': metrice[0]['test_acc'],
            'train_mean_acc': metrice[0]['train_mean_acc'],
            'test_mean_acc': metrice[0]['test_mean_acc'],
        })

        # 如果在一定的训练周期内训练集损失没有继续下降，那么就提前停止训练
        early_stopping(metrice[0]['test_loss'], model)
        if early_stopping.early_stop:
            print('No Improve for test_loss from {} to {}, EarlyStopping.'.format(save_epoch + 1, epoch))
            break
    # 模型训练完毕，计算训练时间,格式化字符串，表示时分秒
    time_end = time.perf_counter()
    # 绘制训练日志的图表
    plot_log(opt)

    # 计算best.pt模型内存
    best_model_path = os.path.join(opt.save_path, 'best.pt')
    best_model = torch.load(best_model_path)['model']
    # 获取模型大小(模型参数数量)
    best_model_size  = getModelSize(best_model)
    print(f"最优模型的内存大小为 {best_model_size} Mb.")

    # 计算训练时间
    time_train = time_end - time_begin
    hours, minutes, seconds = convert_seconds(time_train)
    print(f"训练时间为: {hours}小时{minutes}分钟{seconds}秒")

    # 关闭wandb
    wandb.finish()