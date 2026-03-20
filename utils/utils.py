from sklearn import utils
import torch, itertools, os, time, thop, json, cv2, math, platform, yaml
import torch.nn as nn
import copy
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from math import cos, pi
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score
from prettytable import PrettyTable
from copy import deepcopy
from argparse import Namespace
from PIL import Image
from sklearn.manifold import TSNE
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image
from collections import OrderedDict
from .utils_aug import rand_bbox
from pycm import ConfusionMatrix
from collections import namedtuple
from sklearn.preprocessing import label_binarize

cnames = {
'aliceblue':   '#F0F8FF',
'antiquewhite':   '#FAEBD7',
'aqua':     '#00FFFF',
'aquamarine':   '#7FFFD4',
'azure':    '#F0FFFF',
'beige':    '#F5F5DC',
'bisque':    '#FFE4C4',
'black':    '#000000',
'blanchedalmond':  '#FFEBCD',
'blue':     '#0000FF',
'blueviolet':   '#8A2BE2',
'brown':    '#A52A2A',
'burlywood':   '#DEB887',
'cadetblue':   '#5F9EA0',
'chartreuse':   '#7FFF00',
'chocolate':   '#D2691E',
'coral':    '#FF7F50',
'cornflowerblue':  '#6495ED',
'cornsilk':    '#FFF8DC',
'crimson':    '#DC143C',
'cyan':     '#00FFFF',
'darkblue':    '#00008B',
'darkcyan':    '#008B8B',
'darkgoldenrod':  '#B8860B',
'darkgray':    '#A9A9A9',
'darkgreen':   '#006400',
'darkkhaki':   '#BDB76B',
'darkmagenta':   '#8B008B',
'darkolivegreen':  '#556B2F',
'darkorange':   '#FF8C00',
'darkorchid':   '#9932CC',
'darkred':    '#8B0000',
'darksalmon':   '#E9967A',
'darkseagreen':   '#8FBC8F',
'darkslateblue':  '#483D8B',
'darkslategray':  '#2F4F4F',
'darkturquoise':  '#00CED1',
'darkviolet':   '#9400D3',
'deeppink':    '#FF1493',
'deepskyblue':   '#00BFFF',
'dimgray':    '#696969',
'dodgerblue':   '#1E90FF',
'firebrick':   '#B22222',
'floralwhite':   '#FFFAF0',
'forestgreen':   '#228B22',
'fuchsia':    '#FF00FF',
'gainsboro':   '#DCDCDC',
'ghostwhite':   '#F8F8FF',
'gold':     '#FFD700',
'goldenrod':   '#DAA520',
'gray':     '#808080',
'green':    '#008000',
'greenyellow':   '#ADFF2F',
'honeydew':    '#F0FFF0',
'hotpink':    '#FF69B4',
'indianred':   '#CD5C5C',
'indigo':    '#4B0082',
'ivory':    '#FFFFF0',
'khaki':    '#F0E68C',
'lavender':    '#E6E6FA',
'lavenderblush':  '#FFF0F5',
'lawngreen':   '#7CFC00',
'lemonchiffon':   '#FFFACD',
'lightblue':   '#ADD8E6',
'lightcoral':   '#F08080',
'lightcyan':   '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':   '#90EE90',
'lightgray':   '#D3D3D3',
'lightpink':   '#FFB6C1',
'lightsalmon':   '#FFA07A',
'lightseagreen':  '#20B2AA',
'lightskyblue':   '#87CEFA',
'lightslategray':  '#778899',
'lightsteelblue':  '#B0C4DE',
'lightyellow':   '#FFFFE0',
'lime':     '#00FF00',
'limegreen':   '#32CD32',
'linen':    '#FAF0E6',
'magenta':    '#FF00FF',
'maroon':    '#800000',
'mediumaquamarine':  '#66CDAA',
'mediumblue':   '#0000CD',
'mediumorchid':   '#BA55D3',
'mediumpurple':   '#9370DB',
'mediumseagreen':  '#3CB371',
'mediumslateblue':  '#7B68EE',
'mediumspringgreen': '#00FA9A',
'mediumturquoise':  '#48D1CC',
'mediumvioletred':  '#C71585',
'midnightblue':   '#191970',
'mintcream':   '#F5FFFA',
'mistyrose':   '#FFE4E1',
'moccasin':    '#FFE4B5',
'navajowhite':   '#FFDEAD',
'navy':     '#000080',
'oldlace':    '#FDF5E6',
'olive':    '#808000',
'olivedrab':   '#6B8E23',
'orange':    '#FFA500',
'orangered':   '#FF4500',
'orchid':    '#DA70D6',
'palegoldenrod':  '#EEE8AA',
'palegreen':   '#98FB98',
'paleturquoise':  '#AFEEEE',
'palevioletred':  '#DB7093',
'papayawhip':   '#FFEFD5',
'peachpuff':   '#FFDAB9',
'peru':     '#CD853F',
'pink':     '#FFC0CB',
'plum':     '#DDA0DD',
'powderblue':   '#B0E0E6',
'purple':    '#800080',
'red':     '#FF0000',
'rosybrown':   '#BC8F8F',
'royalblue':   '#4169E1',
'saddlebrown':   '#8B4513',
'salmon':    '#FA8072',
'sandybrown':   '#FAA460',
'seagreen':    '#2E8B57',
'seashell':    '#FFF5EE',
'sienna':    '#A0522D',
'silver':    '#C0C0C0',
'skyblue':    '#87CEEB',
'slateblue':   '#6A5ACD',
'slategray':   '#708090',
'snow':     '#FFFAFA',
'springgreen':   '#00FF7F',
'steelblue':   '#4682B4',
'tan':     '#D2B48C',
'teal':     '#008080',
'thistle':    '#D8BFD8',
'tomato':    '#FF6347',
'turquoise':   '#40E0D0',
'violet':    '#EE82EE',
'wheat':    '#F5DEB3',
'white':    '#FFFFFF',
'whitesmoke':   '#F5F5F5',
'yellow':    '#FFFF00',
'yellowgreen':   '#9ACD32'}

def str2float(data):
    return (0.0 if type(data) is str else data)

def save_model(path, **ckpt):
    torch.save(ckpt, path)

def mixup_data(x, opt, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if opt.mixup == 'mixup':
        mixed_x = lam * x + (1 - lam) * x[index, :]
    elif opt.mixup == 'cutmix':
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = deepcopy(x)
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    else:
        raise 'Unsupported MixUp Methods.'
    return mixed_x

def plot_train_batch(dataset, opt,flag):
    dataset.transform.transforms[-1] = transforms.ToTensor()
    dataloader = iter(torch.utils.data.DataLoader(dataset, 16, shuffle=True))
    for i in range(1, opt.plot_train_batch_count + 1):
        x, _ = next(dataloader)
        if opt.mixup != 'none' and np.random.rand() > 0.5:
            x = mixup_data(x, opt)

        plt.figure(figsize=(10, 10))
        for j in range(16):
            img = transforms.ToPILImage()(x[j])

            plt.subplot(4, 4, 1 + j)
            plt.imshow(np.array(img))
            plt.axis('off')
            plt.title('Sample {}'.format(j + 1))
        plt.tight_layout()
        if flag:
            plt.savefig(r'{}/train_batch{}.png'.format(opt.save_path, i))
        else:
            plt.savefig(r'{}/test_batch{}.png'.format(opt.save_path, i))


import pandas as pd
import os
import matplotlib.pyplot as plt


# 绘制训练曲线 重点！
def plot_log(opt):
    # 读取日志文件
    logs = pd.read_csv(os.path.join(opt.save_path, 'train.log'))

    # 创建画布，大小为10x10
    plt.figure(figsize=(10, 10))

    # 创建子图1，位置为(2, 2, 1)
    plt.subplot(2, 2, 1)

    # 绘制训练损失和验证损失曲线，图例名称分别为'train'和'val'
    plt.plot(logs['loss'], label='train')
    plt.plot(logs['test_loss'], label='val')

    # 尝试绘制kd_loss曲线，如果不存在则跳过
    try:
        plt.plot(logs['kd_loss'], label='kd')
    except:
        pass

    # 添加图例
    plt.legend()

    # 设置标题为'loss'
    plt.title('loss')

    # 设置x轴标签为'epoch'
    plt.xlabel('epoch')

    # 创建子图2，位置为(2, 2, 2)
    plt.subplot(2, 2, 2)

    # 绘制训练准确率和验证准确率曲线，图例名称分别为'train'和'val'
    plt.plot(logs['acc'], label='train')
    plt.plot(logs['test_acc'], label='val')

    # 添加图例
    plt.legend()

    # 设置标题为'acc'
    plt.title('acc')

    # 设置x轴标签为'epoch'
    plt.xlabel('epoch')

    # 创建子图3，位置为(2, 2, 3)
    plt.subplot(2, 2, 3)

    # 绘制训练平均准确率和验证平均准确率曲线，图例名称分别为'train'和'val'
    plt.plot(logs['mean_acc'], label='train')
    plt.plot(logs['test_mean_acc'], label='val')

    # 添加图例
    plt.legend()

    # 设置标题为'mean_acc'
    plt.title('mean_acc')

    # 设置x轴标签为'epoch'
    plt.xlabel('epoch')

    # 调整子图布局
    plt.tight_layout()

    # 保存图像为'iterative_curve.png'，路径为opt.save_path
    plt.savefig(r'{}/iterative_curve.png'.format(opt.save_path))

    # 创建画布，大小为7x5
    plt.figure(figsize=(7, 5))

    # 绘制学习率曲线
    plt.plot(logs['lr'])

    # 设置标题为'learning rate'
    plt.title('learning rate')

    # 设置x轴标签为'epoch'
    plt.xlabel('epoch')

    # 调整子图布局
    plt.tight_layout()

    # 保存图像为'learning_rate_curve.png'，路径为opt.save_path
    plt.savefig(r'{}/learning_rate_curve.png'.format(opt.save_path))


class WarmUpLR:
    def __init__(self, optimizer, opt):
        self.optimizer = optimizer
        self.lr_min = opt.warmup_minlr
        self.lr_max = opt.lr
        self.max_epoch = opt.epoch
        self.current_epoch = 0
        self.lr_scheduler = opt.lr_scheduler(optimizer, **opt.lr_scheduler_params) if opt.lr_scheduler is not None else None
        self.warmup_epoch = int(opt.warmup_ratios * self.max_epoch) if opt.warmup else 0
        if opt.warmup:
            self.step()

    def step(self):
        self.adjust_lr()
        self.current_epoch += 1

    def adjust_lr(self):
        if self.current_epoch <= self.warmup_epoch and self.warmup_epoch != 0:
            lr = (self.lr_max - self.lr_min) * (self.current_epoch / self.warmup_epoch) + self.lr_min
        else:
            if self.lr_scheduler:
                self.lr_scheduler.step()
                return
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * (
                        1 + cos(pi * (self.current_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch))) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    def state_dict(self):
        return {
            'lr_min': self.lr_min,
            'lr_max': self.lr_max,
            'max_epoch': self.max_epoch,
            'current_epoch': self.current_epoch,
            'warmup_epoch': self.warmup_epoch,
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.lr_min = state_dict['lr_min']
        self.lr_max = state_dict['lr_max']
        self.max_epoch = state_dict['max_epoch']
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epoch = state_dict['warmup_epoch']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

def show_config(opt):
    table = PrettyTable()
    table.title = 'Configurations'
    table.field_names = ['params', 'values']
    opt = vars(opt)
    for key in opt:
        table.add_row([str(key), str(opt[key]).replace('\n', '')])
    print(table)

    if not opt['resume']:
        for keys in opt.keys():
            if type(opt[keys]) is not str:
                opt[keys] = str(opt[keys]).replace('\n', '')
            else:
                opt[keys] = opt[keys].replace('\n', '')

        with open(os.path.join(opt['save_path'], 'param.yaml'), 'w+') as f:
            # f.write(json.dumps(opt, indent=4, separators={':', ','}))
            yaml.dump(opt, f)


# 混淆矩阵绘制图 我把normalize改成了True，因为数据集划分不能完全按照比例划分，所以得用分数，不能用数量
def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues, name=''):
    plt.figure(figsize=(min(len(classes), 30), min(len(classes), 30)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    trained_classes = classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name + title, fontsize=min(len(classes), 30))  # title font size
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=min(len(classes), 30)) # X tricks font size
    plt.yticks(tick_marks, classes, fontsize=min(len(classes), 30)) # Y tricks font size
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=min(len(classes), 30)) # confusion_matrix font size
    plt.ylabel('真实标签', fontsize=min(len(classes), 30)) # True label font size
    plt.xlabel('预测标签', fontsize=min(len(classes), 30)) # Predicted label font size
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150)
    # plt.show()

def save_confusion_matrix(cm, classes, save_path):
    str_arr = []
    for class_, cm_ in zip(classes, cm):
        str_arr.append('{},{}'.format(class_, ','.join(list(map(lambda x:'{:.4f}'.format(x), list(cm_))))))
    str_arr.append(' ,{}'.format(','.join(classes)))

    with open(os.path.join(save_path, 'confusion_matrix.csv'), 'w+') as f:
        f.write('\n'.join(str_arr))

def cal_cm(y_true, y_pred, CLASS_NUM):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(CLASS_NUM)))
    return cm

class Train_Metrice:
    def __init__(self, class_num):
        self.train_cm = np.zeros((class_num, class_num))
        self.test_cm = np.zeros((class_num, class_num))
        self.class_num = class_num

        self.train_loss = []
        self.test_loss = []
        self.train_kd_loss = []
    
    def update_y(self, y_true, y_pred, isTest=False):
        if isTest:
            self.test_cm += cal_cm(y_true, y_pred, self.class_num)
        else:
            self.train_cm += cal_cm(y_true, y_pred, self.class_num)
    
    def update_loss(self, loss, isTest=False, isKd=False):
        if isKd:
            self.train_kd_loss.append(loss)
        else:
            if isTest:
                self.test_loss.append(loss)
            else:
                self.train_loss.append(loss)

    def get(self):
        result = {}
        result['train_loss'] = np.mean(self.train_loss)
        result['test_loss'] = np.mean(self.test_loss)
        if len(self.train_kd_loss) != 0:
            result['train_kd_loss'] = np.mean(self.train_kd_loss)
        result['train_acc'] = np.diag(self.train_cm).sum() / (self.train_cm.sum() + 1e-7)
        result['test_acc'] = np.diag(self.test_cm).sum() / (self.test_cm.sum() + 1e-7)
        result['train_mean_acc'] = np.diag(self.train_cm.astype('float') / self.train_cm.sum(axis=1)[:, np.newaxis]).mean()
        result['test_mean_acc'] = np.diag(self.test_cm.astype('float') / self.test_cm.sum(axis=1)[:, np.newaxis]).mean()

        # table = PrettyTable()
        # table.title = 'Metrice'
        if 'train_kd_loss' in result:
            cols_name = ['train_loss', 'train_kd_loss', 'train_acc', 'train_mean_acc', 'test_loss', 'test_acc', 'test_mean_acc']
        else:
            cols_name = ['train_loss', 'train_acc', 'train_mean_acc', 'test_loss', 'test_acc', 'test_mean_acc']
        # table.field_names = cols_name
        # table.add_row(['{:.5f}'.format(result[i]) for i in cols_name])

        return result, ','.join(['{:.6f}'.format(result[i]) for i in cols_name])

class Test_Metrice:
    def __init__(self, y_true, y_pred, class_num):
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_num = class_num
        self.result = {i:{} for i in range(self.class_num)}
        self.metrice = ['PPV', 'TPR', 'AUC', 'AUPR', 'F05', 'F1', 'F2']
        self.metrice_name = ['Precision', 'Recall', 'AUC', 'AUPR', 'F0.5', 'F1', 'F2', 'ACC']
    
    def __call__(self):
        cm = ConfusionMatrix(self.y_true, self.y_pred)
        for j in range(len(self.metrice)):
            for i in range(self.class_num):
                self.result[i][self.metrice_name[j]] = str2float(eval('cm.{}'.format(self.metrice[j]))[i])

        return self.result, cm


# 绘制分类roc曲线
def multiclass_roc_auc(y_true,y_pred,label,save_path):

    # 将标签二值化
    y_true_bin = label_binarize(y_true, classes= list(range(len(label))))
    y_pred_bin = label_binarize(y_pred, classes= list(range(len(label))))
    # 计算每个类别的ROC和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(label)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算Micro-ROC和AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # 计算Macro-ROC和AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(label))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(label)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(label)
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    # 绘制每个类别的ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], label=f'Class {label[i]} (AUC = {roc_auc[i]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Multiclass_ROC_Curve.png'))
    # 绘制Micro-ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-Average ROC (AUC = {roc_auc_micro:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Micro_ROC_Curve.png'))
    # 绘制Macro-ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_macro, tpr_macro, label=f'Macro-Average ROC (AUC = {roc_auc_macro:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Macro_ROC_Curve.png'))
    # 将上面三张图的曲线叠加到一张图里面
    plt.figure(figsize=(8, 6))
    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta']
    # 随机曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
    # 每个类别的 ROC 曲线
    for i, color in zip(range(len(label)), colors):
        plt.plot(fpr[i], tpr[i], color=color, label=f'Class {label[i]} (AUC = {roc_auc[i]:.2f})')
    # Micro-Average ROC 曲线
    plt.plot(fpr_micro, tpr_micro, linestyle='--', color='navy', label=f'Micro-Average ROC (AUC = {roc_auc_micro:.2f})')
    # Macro-Average ROC 曲线
    plt.plot(fpr_macro, tpr_macro, linestyle='--', color='purple', label=f'Macro-Average ROC (AUC = {roc_auc_macro:.2f})')
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve(ALL)')
    plt.savefig(os.path.join(save_path, 'Multiclass_ROC_Curve_ALL.png'))
    # 将roc相关数据存储到一个csv文件中，以便后面用matlab绘图
    roc_data = pd.DataFrame()
    # 将Macro-ROC曲线数据和AUC值添加到DataFrame中
    roc_data['Macro-Average FPR'] = fpr_macro
    roc_data['Macro-Average TPR'] = tpr_macro
    roc_data.at[0, 'Macro-Average AUC'] = roc_auc_macro

    # 判断fpr_macro和fprr_micro的长度
    num_ap=len(fpr_macro)-len(fpr_micro)

    # 将Micro-ROC曲线数据和AUC值添加到DataFrame中
    roc_data['Micro-Average FPR'] = np.append(fpr_micro,[0]*num_ap)
    roc_data['Micro-Average TPR'] = np.append(tpr_micro,[0]*num_ap)
    roc_data.at[0, 'Micro-Average AUC'] = roc_auc_micro
    for i in range(len(label)):
        # 将每个类别的ROC曲线数据添加到DataFrame
        roc_data[f'Class {label[i]} FPR'] = np.append(fpr[i],[0]*num_ap)
        roc_data[f'Class {label[i]} TPR'] = np.append(tpr[i],[0]*num_ap)
        # 将每个类别的AUC值添加到DataFrame中
        roc_data.at[0, f'Class {label[i]} AUC'] = roc_auc[i]

    # 将DataFrame保存为CSV文件
    roc_data.to_csv(os.path.join(save_path, 'roc.csv'), index=False,encoding='gbk')

    # 返回marco_auc，取5位小数
    return round(roc_auc_macro, 5)

# 绘制分类pr曲线
def multiclass_pr_curve(y_true, y_pred, label, save_path):
    # 将标签二值化
    y_true_bin = label_binarize(y_true, classes= list(range(len(label))))
    y_pred_bin = label_binarize(y_pred, classes= list(range(len(label))))
    # 计算每个类别的PR曲线和AP值
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(label)):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])

    # 计算Macro-AP值,即mAP
    mAP = average_precision_score(y_true_bin, y_pred_bin, average="macro")
    # 绘制每个类别的PR曲线
    plt.figure(figsize=(8, 6))
    for i in range(len(label)):
        plt.plot(recall[i], precision[i], label=f'Class {label[i]} (AP = {average_precision[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multiclass PR Curve')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'Multiclass_PR_Curve.png'))

    # 将PR相关数据存储到一个csv文件中，以便后面用matlab绘图
    pr_data = pd.DataFrame()
    for i in range(len(label)):
        # 将每个类别的PR曲线数据添加到DataFrame中
        pr_data[f'Class {label[i]} Precision'] = precision[i]
        pr_data[f'Class {label[i]} Recall'] = recall[i]
        # 将每个类别的AP值添加到DataFrame中
        pr_data.at[0, f'Class {label[i]} AP'] = average_precision[i]
    # 将mAP值添加到DataFrame中
    pr_data.at[0, 'mAP'] = mAP
    # 将DataFrame保存为CSV文件
    pr_data.to_csv(os.path.join(save_path, 'PR.csv'), index=False,encoding='gbk')
    # 返回mAP，取5位小数
    return round(mAP, 5)

# 计算多分类g-mean指标，它是各个类别的准确率的几何平均值
# g-mean的优点是它能够平衡各个类别的性能，避免因为某些类别的样本数量过大或过小而导致的偏差。g-mean越高，表示多分类模型的性能越好。
def multiclass_g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    accuracy_product = 1.0
    for i in range(num_classes):
        tp = cm[i, i]
        actual_positive = cm[i, :].sum()
        predicted_positive = cm[:, i].sum()
        accuracy = tp / (actual_positive + predicted_positive - tp) if actual_positive + predicted_positive - tp != 0 else 0
        accuracy_product *= accuracy
    # 多分类g-mean=（各类别准确率相乘）^(1/类别数)
    G_mean = accuracy_product ** (1. / num_classes)
    return round(G_mean,5)


# 添加了inference_time,fps，要放到训练日志里面
def classification_metrice(y_true, y_pred, class_num, label, save_path,inference_time,fps):
    metrice = Test_Metrice(y_true, y_pred, class_num)
    class_report, cm = metrice()
    class_pa = np.diag(cm.to_array(normalized=True)) # mean class accuracy
    if class_num <= 50:
        # 绘制混淆矩阵
        plot_confusion_matrix(cm.to_array(), label, save_path)
    save_confusion_matrix(cm.to_array(normalized=True), label, save_path)

    # 绘制roc曲线
    roc_auc_macro=multiclass_roc_auc(y_true,y_pred,label,save_path)
    # 计算多分类g-mean
    G_mean = multiclass_g_mean(y_true, y_pred)
    # 绘制pr曲线
    mAP=multiclass_pr_curve(y_true, y_pred, label, save_path)

    table1_cols_name = ['class'] + metrice.metrice_name
    table1 = PrettyTable()
    table1.title = 'Per Class'
    table1.field_names = table1_cols_name
    with open(os.path.join(save_path, 'perclass_result.csv'), 'w+', encoding='gbk') as f:
        f.write(','.join(table1_cols_name) + '\n')
        for i in range(class_num):
            table1.add_row([label[i]] + ['{:.5f}'.format(class_report[i][j]) for j in table1_cols_name[1:-1]] + ['{:.5f}'.format(class_pa[i])])
            f.write(','.join([label[i]] + ['{:.5f}'.format(class_report[i][j]) for j in table1_cols_name[1:-1]] + ['{:.5f}'.format(class_pa[i])]) + '\n')
    print(table1)
    
    table2_cols_name = [
                        'Accuracy', 'Mean_Accuracy', 'F1_Macro',
                        'AUC_Macro','G-mean','mAP','Kappa', 
                        'Precision_Micro', 'Recall_Micro', 
                        'F1_Micro', 'Precision_Macro', 'Recall_Macro', ]
    table2 = PrettyTable()
    table2.title = 'Overall'
    table2.field_names = table2_cols_name
    with open(os.path.join(save_path, 'overall_result.csv'), 'w+', encoding='utf-8') as f:
        data = [
                '{:.5f}'.format(str2float(cm.Overall_ACC)),
                '{:.5f}'.format(np.mean(class_pa)), # 计算mean_accuracy
                '{:.5f}'.format(str2float(cm.F1_Macro)),
                '{:.5f}'.format(roc_auc_macro),
                '{:.5f}'.format(G_mean),
                '{:.5f}'.format(mAP),
                '{:.5f}'.format(str2float(cm.Kappa)),
                '{:.5f}'.format(str2float(cm.PPV_Micro)),
                '{:.5f}'.format(str2float(cm.TPR_Micro)),
                '{:.5f}'.format(str2float(cm.F1_Micro)),
                '{:.5f}'.format(str2float(cm.PPV_Macro)),
                '{:.5f}'.format(str2float(cm.TPR_Macro)),
        ]

        table2.add_row(data)

        f.write(','.join(table2_cols_name) + '\n')
        f.write(','.join(data))
    print(table2)

    with open(os.path.join(save_path, 'result.txt'), 'w+', encoding='utf-8') as f:
        f.write(str(table1))
        f.write('\n')
        f.write(str(table2))

def update_opt(a, b):
    b = vars(b)
    for key in b:
        setattr(a, str(key), b[key])
    return a


# 优化器设置
def setting_optimizer(opt, model):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
    
    name = opt.optimizer
    if name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=opt.lr, betas=(opt.momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=opt.lr, momentum=opt.momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=opt.lr, momentum=opt.momentum, nesterov=True)
    
    optimizer.add_param_group({'params': g[0], 'weight_decay': opt.weight_decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)

    return optimizer

def check_batch_size(model, imgsz=640, amp=True):
    # Check YOLOv5 training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size

def time_sync(): # 注意这个函数，确保在进行性能测试的时候，不会因为异步操作导致时间不准确，用在了计算flops的函数里面
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def profile(input, ops, n=10, device=None):
    """
    这个函数用于计算并打印模型的性能指标，包括参数数量、浮点运算次数（GFLOPs）、GPU内存使用量、前向传播时间、反向传播时间、输入输出的形状。

    参数:
    input: 输入数据，可以是一个列表，也可以是一个单独的输入。
    ops: 要进行性能分析的操作或模型，可以是一个列表，也可以是一个单独的操作或模型。
    n: 运行操作或模型的次数，用于计算平均前向传播和反向传播时间。
    device: 运行操作或模型的设备，如果没有指定，则使用默认设备。

    返回值:
    results: 一个列表，包含了每个操作或模型的性能指标。
    """
    # 初始化结果列表
    results = []
    # 打印表头
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    # 遍历输入数据
    for x in input if isinstance(input, list) else [input]:
        # 将输入数据移动到指定设备
        x = x.to(device)
        # 设置输入数据需要计算梯度
        x.requires_grad = True
        # 遍历操作或模型
        for m in ops if isinstance(ops, list) else [ops]:
            # 如果操作或模型有 'to' 方法，将其移动到指定设备
            m = m.to(device) if hasattr(m, 'to') else m
            # 如果操作或模型有 'half' 方法，并且输入数据是半精度浮点数，将操作或模型转换为半精度浮点数
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            # 初始化前向传播时间、反向传播时间和临时时间列表
            tf, tb, t = 0, 0, [0, 0, 0]
            try:
                # 计算操作或模型的浮点运算次数（GFLOPs）
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2
            except Exception:
                # 如果计算失败，设置浮点运算次数为 0
                flops = 0

            try:
                # 运行操作或模型 n 次
                for _ in range(n):
                    # 记录前向传播开始时间
                    t[0] = time_sync()
                    # 进行前向传播
                    y = m(x)
                    # 记录前向传播结束时间
                    t[1] = time_sync()
                    try:
                        # 进行反向传播
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        # 记录反向传播结束时间
                        t[2] = time_sync()
                    except Exception as e:  # 没有反向传播方法
                        # 打印错误信息（用于调试）
                        print(e)
                        # 设置反向传播结束时间为 NaN
                        t[2] = float('nan')
                    # 计算并累加前向传播时间
                    tf += (t[1] - t[0]) * 1000 / n
                    # 计算并累加反向传播时间
                    tb += (t[2] - t[1]) * 1000 / n
                # 获取 GPU 内存使用量（GB）
                mem = torch.cuda.memory_reserved(device) / 1E9 if torch.cuda.is_available() else 0
                # 获取输入和输出的形状
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))
                # 获取参数数量
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0
                # 打印性能指标
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                # 将性能指标添加到结果列表
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                # 打印错误信息
                print(e)
                # 将 None 添加到结果列表
                results.append(None)
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
    # 返回结果列表
    return results
def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Check device
    prefix = 'AutoBatch: '
    print(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        print(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    print(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        print(f'{prefix}{e}')

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1:  # zero or negative batch size
        raise(f'{prefix}WARNING: ⚠️ CUDA anomaly detected, recommend restart environment and retry command.') # raise exception if batch size < 1

    fraction = np.polyval(p, b) / t  # actual fraction predicted
    print(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b

class Metrice_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pil_img = Image.open(self.dataset.imgs[index][0])
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        pil_img = self.dataset.transform(pil_img)
        return pil_img, self.dataset.imgs[index][1], self.dataset.imgs[index][0]

def visual_predictions(y_true, y_pred, y_score, path, label, save_path):
    true_ids = (y_true == y_pred)
    with open(os.path.join(save_path, 'correct.csv'), 'w+') as f:
        f.write('path,true_label,pred_label,pred_score\n')
        f.write('\n'.join(['{},{},{},{:.4f}'.format(i, label[j], label[k], z) for i, j, k, z in zip(path[true_ids], y_true[true_ids], y_pred[true_ids], y_score[true_ids])]))

    with open(os.path.join(save_path, 'incorrect.csv'), 'w+') as f:
        f.write('path,true_label,pred_label,pred_score\n')
        f.write('\n'.join(['{},{},{},{:.4f}'.format(i, label[j], label[k], z) for i, j, k, z in zip(path[~true_ids], y_true[~true_ids], y_pred[~true_ids], y_score[~true_ids])]))
    
def visual_tsne(feature, y_true, path, labels, save_path):
    # 这个函数可视化的是模型网络最后一层特征的tsne图
    color_name_list = list(sorted(cnames.keys()))
    np.random.shuffle(color_name_list)
    tsne = TSNE(n_components=2)
    feature_tsne = tsne.fit_transform(feature)

    if len(labels) <= len(color_name_list):
        plt.figure(figsize=(8, 8))
        for idx, i in enumerate(labels):
            plt.scatter(feature_tsne[y_true == idx, 0], feature_tsne[y_true == idx, 1], label=i, c=cnames[color_name_list[idx]])
        plt.legend(loc='best')
        plt.title('Tsne 可视化图')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'tsne.png'))

    with open(os.path.join(save_path, 'tsne.csv'), 'w+') as f:
        f.write('path,label,tsne_x,tsne_y\n')
        f.write('\n'.join(['{},{},{:.0f},{:.0f}'.format(i, labels[j], k[0], k[1]) for i, j, k in zip(path, y_true, feature_tsne)]))


def predict_single_image(path, model, test_transform, DEVICE, half=False):
    pil_img = Image.open(path)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    tensor_img = test_transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor_img = (tensor_img.half() if (half and torch.cuda.is_available()) else tensor_img)
    with torch.inference_mode():
        if len(tensor_img.shape) == 5:
            tensor_img = tensor_img.reshape((tensor_img.size(0) * tensor_img.size(1), tensor_img.size(2), tensor_img.size(3), tensor_img.size(4)))
            output = model(tensor_img).mean(0)
        else:
            output = model(tensor_img)[0]

    try:
        pred_result = torch.softmax(output, 0)
    except:
        pred_result = torch.softmax(torch.from_numpy(output), 0) # using torch.softmax will faster than numpy
    return int(pred_result.argmax()), pred_result

class cam_visual:
    def __init__(self, model, test_transform, DEVICE, opt):
        self.test_transform = test_transform
        self.DEVICE = DEVICE
        self.opt = opt

        # 创建一个类激活映射（CAM）模型
        # eval(opt.cam_type) 通过执行字符串 opt.cam_type 获取 CAM 模型的类
        # model=model 将当前模型作为参数传递给 CAM 模型
        # target_layers=model.cam_layer() 调用模型的 cam_layer 方法，获取需要进行类激活映射的层
        # use_cuda=(DEVICE.type != 'cpu') 判断是否在 GPU 上运行模型，如果设备类型不是 'cpu'，则在 GPU 上运行模型
        self.cam_model = eval(opt.cam_type)(model=model, target_layers=model.cam_layer(), use_cuda=(DEVICE.type != 'cpu'))
        self.model = model
    
    def __call__(self, path):
        pil_img = Image.open(path).convert("RGB")
        tensor_img = self.test_transform(pil_img).unsqueeze(0).to(self.DEVICE)
        
        if len(tensor_img.shape) == 5:
            grayscale_cam_list = [self.cam_model(input_tensor=tensor_img[:, i], targets=None) for i in range(tensor_img.size(1))]
        else:
            grayscale_cam_list = [self.cam_model(input_tensor=tensor_img, targets=None)]
        
        grayscale_cam = np.concatenate(grayscale_cam_list, 0).mean(0)
        grayscale_cam = cv2.resize(grayscale_cam, pil_img.size)
        pil_img_np = np.array(pil_img, dtype=np.float32) / 255.0
        return Image.fromarray(show_cam_on_image(pil_img_np, grayscale_cam, use_rgb=True))

def load_weights_from_state_dict(model, state_dict):
    model_dict = model.state_dict()
    weight_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if np.shape(model_dict[k]) == np.shape(v):
                weight_dict[k] = v
    unload_keys = list(set(model_dict.keys()).difference(set(weight_dict.keys())))
    if len(unload_keys) == 0:
        print('all keys is loading.')
    elif len(unload_keys) <= 50:
        print('unload_keys:{} unload_keys_len:{} unload_keys/weight_keys:{:.3f}%'.format(','.join(unload_keys), len(unload_keys), len(unload_keys) / len(model_dict) * 100))
    else:
        print('unload_keys:{}.... unload_keys_len:{} unload_keys/weight_keys:{:.3f}%'.format(','.join(unload_keys[:50]), len(unload_keys), len(unload_keys) / len(model_dict) * 100))
    pretrained_dict = weight_dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def load_weights(model, opt):
    if opt.weight:
        if not os.path.exists(opt.weight):
            print('opt.weight not found, skipping...')
        else:
            print('found weight in {}, loading...'.format(opt.weight))
            state_dict = torch.load(opt.weight)
            if type(state_dict) is dict:
                try:
                    state_dict = state_dict['model'].state_dict()
                except:
                    pass
            elif not (state_dict is OrderedDict):
                state_dict = state_dict.state_dict()
            model = load_weights_from_state_dict(model, state_dict)
    return model

def get_channels(model, opt):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.rand((1, opt.image_channel, opt.image_size, opt.image_size)).to(DEVICE)
    model.eval()
    model.to(DEVICE)
    return model.forward_features(inputs, True)[0][-1].size(1)

def dict_to_PrettyTable(data, name):
    data_keys = list(data.keys())
    table = PrettyTable()
    table.title = name
    table.field_names = data_keys
    table.add_row(['{:.5f}'.format(data[i]) for i in data_keys])
    return str(table)

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model




def _fuse_conv_bn_fasternet(conv: nn.Module, bn: nn.Module) -> nn.Module:
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv


def fuse_conv_bn_fasternet(module: nn.Module) -> nn.Module:
    """Recursively fuse conv and bn in a module.

    During inference, the functionary of batch norm layers is turned off
    but only the mean and var alone channels are used, which exposes the
    chance to fuse it with the preceding conv layers to save computations and
    simplify network structures.

    Args:
        module (nn.Module): Module to be fused.

    Returns:
        nn.Module: Fused module.
    """
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn_fasternet(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn_fasternet(child)
    return module


class Model_Inference:
    def __init__(self, device, opt):
        self.opt = opt
        self.device = device

        if self.opt.model_type == 'torch':
            # 从指定路径opt.save_path+best.pt加载模型
            ckpt = torch.load(os.path.join(opt.save_path, 'best.pt'))
            # 从检查点中加载模型，将模型数据转化为浮点型
            self.model = ckpt['model'].float()
            if 'fasternet' in opt.save_path:
                self.model = fuse_conv_bn_fasternet(self.model)
            else:
            # 模型融合，加快模型运行效率
                model_fuse(self.model)
            # 将模型转移到指定设备
            self.model = (self.model.half() if opt.half else self.model)
            self.model.to(self.device)
            # 设置模型为评估模式
            self.model.eval()
            # 如果模型名字中含有'mobileone',这个模型部署的时候需要重新参数化
            if 'mobileone' in opt.save_path:
                self.model = reparameterize_model(self.model)

        elif self.opt.model_type == 'onnx':
            import onnx, onnxruntime
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.model = onnxruntime.InferenceSession(os.path.join(opt.save_path, 'best.onnx'), providers=providers)
        elif self.opt.model_type == 'torchscript':
            self.model = torch.jit.load(os.path.join(opt.save_path, 'best.ts'))
            self.model = (self.model.half() if opt.half else self.model)
            self.model.to(self.device)
            self.model.eval()
        elif self.opt.model_type == 'tensorrt':
            import tensorrt as trt
            if device.type == 'cpu':
                raise RuntimeError('Tensorrt not support CPU Inference.')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger()
            with open(os.path.join(opt.save_path, 'best.engine'), 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            fp16 = False  # default updated below
            dynamic = False
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                if model.binding_is_input(index):
                    if -1 in tuple(model.get_binding_shape(index)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(index, tuple(model.get_profile_shape(0, index)[2]))
                    if dtype == np.float16:
                        fp16 = True
                shape = tuple(context.get_binding_shape(index))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.bindings = bindings
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            self.batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
            self.model = model
            self.fp16 = fp16
            self.dynamic = dynamic
            self.context = context
    
    def __call__(self, inputs):
        if self.opt.model_type == 'torch':
            with torch.inference_mode():
                return self.model(inputs)
        elif self.opt.model_type == 'onnx':
            inputs = inputs.cpu().numpy().astype(np.float16 if '16' in self.model.get_inputs()[0].type else np.float32)
            return self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: inputs})[0]
        elif self.opt.model_type == 'torchscript':
            return self.model(inputs)
        elif self.opt.model_type == 'tensorrt':
            if self.fp16:
                inputs = inputs.half()
            if self.dynamic and inputs.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, inputs.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=inputs.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert inputs.shape == s, f"input size {inputs.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(inputs.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
            return y
    
    def forward_features(self, inputs):
        try:
            return self.model.forward_features(inputs)
        except:
            raise 'this model is not a torch model.'
    
    def cam_layer(self):
        try:
            return self.model.cam_layer()
        except:
            raise 'this model is not a torch model.'


def select_device(device='', batch_size=0):
    # 将设备名称转换为小写字符串，并删除'cuda:'和'none'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  
    # 检查是否选择了CPU
    cpu = device == 'cpu'
    if cpu:
        # 如果选择了CPU，设置环境变量'CUDA_VISIBLE_DEVICES'为'-1'，使CUDA设备不可见
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        # 如果选择了CUDA设备，设置环境变量'CUDA_VISIBLE_DEVICES'为设备编号
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # 检查CUDA设备是否可用，并且设备数量不少于请求的设备数量
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    # 初始化打印字符串，包含Python和PyTorch的版本信息
    print_str = f'Image-Classifier Python-{platform.python_version()} Torch-{torch.__version__} '
    if not cpu and torch.cuda.is_available():
        # 如果选择了CUDA设备，获取设备编号列表
        devices = device.split(',') if device else '0'
        n = len(devices)  # 设备数量
        if n > 1 and batch_size > 0:  # 检查batch_size是否可以被设备数量整除
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(print_str)
        for i, d in enumerate(devices):
            # 获取每个设备的属性，并添加到打印字符串中
            p = torch.cuda.get_device_properties(i)
            print_str += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = 'cuda:0'
    else:
        # 如果选择了CPU，添加到打印字符串中
        print_str += 'CPU'
        arg = 'cpu'
    # 打印设备信息
    print(print_str)
    # 返回PyTorch设备对象
    return torch.device(arg)

def fuse_conv_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    # 此处没有考虑到dilation参数，导致混合后的卷积层的dilation参数为1
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
            dilation=conv.dilation,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

def model_fuse(model):
    before_fuse_layers = len(getLayers(model))
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    print(f'model fuse... {before_fuse_layers} layers to {len(getLayers(model))} layers')

def getLayers(model):
    """
    get each layer's name and its module
    :param model:
    :return: each layer's name and its module
    """
    layers = []
 
    def unfoldLayer(model):
        """
        unfold each layer
        :param model: the given model or a single layer
        :param root: root name
        :return:
        """
 
        # get all layers of the model
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)
 
            # if current layer contains sublayers, add current layer name on its sublayers
            if sublayer_num == 0:
                layers.append(module)
            # if current layer contains sublayers, unfold them
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)
 
    unfoldLayer(model)
    return layers