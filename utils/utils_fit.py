import torch, tqdm
import numpy as np
from copy import deepcopy
from .utils_aug import mixup_data, mixup_criterion
from .utils import Train_Metrice
import time
from mainweb_globalValue import stop_training_flag
def fitting(model, ema, loss, optimizer, train_dataset, test_dataset, CLASS_NUM, DEVICE, scaler, show_thing, opt):
    global stop_training_flag
    # 设置模型为训练模式
    model.train()
    # 初始化训练指标
    metrice = Train_Metrice(CLASS_NUM)
    # 遍历训练数据集
    for x, y in tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing)):
        # 将数据和标签移动到设备上，并转换为适当的数据类型
        x, y = x.to(DEVICE).float(), y.to(DEVICE).long()
        # 如果中止训练标志为True，则中止训练
        if stop_training_flag:
            return None
        # 开启混合精度训练
        with torch.cuda.amp.autocast(opt.amp):
            # 判断是否使用R-Drop和Mixup
            if opt.rdrop:
                if opt.mixup != 'none' and np.random.rand() > 0.5:
                    # 使用Mixup
                    x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                    pred = model(x_mixup)
                    pred2 = model(x_mixup)
                    l = mixup_criterion(loss, [pred, pred2], y_a, y_b, lam)
                    pred = model(x)
                else:
                    pred = model(x)
                    pred2 = model(x)
                    l = loss([pred, pred2], y)
            else:
                if opt.mixup != 'none' and np.random.rand() > 0.5:
                    # 使用Mixup
                    x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                    pred = model(x_mixup)
                    l = mixup_criterion(loss, pred, y_a, y_b, lam)
                    pred = model(x)
                else:
                    # # 不使用Mixup
                    # pred = model(x)
                    if 'fdt_capsnet' in opt.save_path :
                        # 胶囊网络需要用到重构损失
                        pred,reconstruction = model(x)
                        l = loss(x,y,pred,reconstruction)
                    else:
                        pred = model(x)
                        # 计算一个minibatch的损失
                        l = loss(pred, y)

        # 更新训练指标
        metrice.update_loss(float(l.data))
        metrice.update_y(y, pred)
        # torch.autograd.set_detect_anomaly(True)
        # 反向传播和优化
        # with torch.autograd.detect_anomaly():
        scaler.scale(l).backward()

        # 在反向传播之后，但在优化器步骤之前
        # 这个裁剪梯度，max_norm调到1e-3这种，根本学习不到新参数，要么取值为1
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # 更新EMA模型
        if ema:
            ema.update(model)

    # 获取用于评估的模型
    if ema:
        model_eval = ema.ema
    else:
        model_eval = model.eval()
    # 开启推理模式
    with torch.inference_mode():
        # 遍历测试数据集
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

            # 开启混合精度训练
            with torch.cuda.amp.autocast(opt.amp):
                if opt.test_tta:
                    # 使用TTA
                    bs, ncrops, c, h, w = x.size()
                    pred = model_eval(x.view(-1, c, h, w))
                    pred = pred.view(bs, ncrops, -1).mean(1)
                    l = loss(pred, y)
                else:
                    if 'fdt_capsnet' in opt.save_path :
                        # 胶囊网络需要用到重构损失
                        pred,reconstruction = model_eval(x)
                        l = loss(x,y,pred,reconstruction)
                    else:
                        pred = model_eval(x)
                        l = loss(pred, y)

            metrice.update_loss(float(l.data), isTest=True)
            metrice.update_y(y, pred, isTest=True)

    return metrice.get()


def fitting_distill(teacher_model, student_model, ema, loss, kd_loss, optimizer, train_dataset, test_dataset, CLASS_NUM,
                    DEVICE, scaler, show_thing, opt):
    student_model.train()
    metrice = Train_Metrice(CLASS_NUM)
    for x, y in tqdm.tqdm(train_dataset, desc='{} Train Stage'.format(show_thing)):
        x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

        with torch.cuda.amp.autocast(opt.amp):
            if opt.mixup != 'none' and np.random.rand() > 0.5:
                x_mixup, y_a, y_b, lam = mixup_data(x, y, opt)
                s_features, s_features_fc, s_pred = student_model(x_mixup, need_fea=True)
                t_features, t_features_fc, t_pred = teacher_model(x_mixup, need_fea=True)
                l = mixup_criterion(loss, s_pred, y_a, y_b, lam)
                pred = student_model(x)
            else:
                s_features, s_features_fc, s_pred = student_model(x, need_fea=True)
                t_features, t_features_fc, t_pred = teacher_model(x, need_fea=True)
                l = loss(s_pred, y)
            if str(kd_loss) in ['SoftTarget']:
                kd_l = kd_loss(s_pred, t_pred)
            elif str(kd_loss) in ['MGD']:
                kd_l = kd_loss(s_features[-1], t_features[-1])
            elif str(kd_loss) in ['SP']:
                kd_l = kd_loss(s_features[2], t_features[2]) + kd_loss(s_features[3], t_features[3])
            elif str(kd_loss) in ['AT']:
                kd_l = kd_loss(s_features[2], t_features[2]) + kd_loss(s_features[3], t_features[3])
                    
            if str(kd_loss) in ['SoftTarget', 'SP', 'MGD']:
                kd_l *= (opt.kd_ratio / (1 - opt.kd_ratio)) if opt.kd_ratio < 1 else opt.kd_ratio
            elif str(kd_loss) in ['AT']:
                kd_l *= opt.kd_ratio

        metrice.update_loss(float(l.data))
        metrice.update_loss(float(kd_l.data), isKd=True)
        if opt.mixup != 'none':
            metrice.update_y(y, pred)
        else:
            metrice.update_y(y, s_pred)

        scaler.scale(l + kd_l).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ema:
            ema.update(student_model)

    if ema:
        model_eval = ema.ema
    else:
        model_eval = student_model.eval()
    with torch.inference_mode():
        for x, y in tqdm.tqdm(test_dataset, desc='{} Test Stage'.format(show_thing)):
            x, y = x.to(DEVICE).float(), y.to(DEVICE).long()

            with torch.cuda.amp.autocast(opt.amp):
                if opt.test_tta:
                    bs, ncrops, c, h, w = x.size()
                    pred = model_eval(x.view(-1, c, h, w))
                    pred = pred.view(bs, ncrops, -1).mean(1)
                    l = loss(pred, y)
                else:
                    pred = model_eval(x)
                    l = loss(pred, y)

            metrice.update_loss(float(l.data), isTest=True)
            metrice.update_y(y, pred, isTest=True)

    return metrice.get()


# 编写早停函数
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping ！")
        else:
            self.best_score = score
            self.counter = 0



