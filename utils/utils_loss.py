import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['PolyLoss', 'CrossEntropyLoss', 'FocalLoss', 'RDropLoss','FocalMarginLoss','PSoftmaxLoss','CapsuleLoss']

class PolyLoss(torch.nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    <https://arxiv.org/abs/2204.12511>
    """
    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor = None, epsilon=2.0):
        super().__init__()
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, outputs, targets):
        ce = F.cross_entropy(outputs, targets, label_smoothing=self.label_smoothing, weight=self.weight)
        pt = F.one_hot(targets, outputs.size()[1]) * F.softmax(outputs, 1)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=1))).mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, weight: torch.Tensor = None):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(input, target)

class FocalLoss(nn.Module):
    def __init__(self, label_smoothing:float = 0.0, weight: torch.Tensor = None, gamma:float = 3.0):
        super(FocalLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 对目标标签进行one-hot编码
        target_onehot = F.one_hot(target, num_classes=input.size(1))
        # 对one-hot编码的标签应用标签平滑，将标签的0值替换为label_smoothing/(input.size(1)-1)，将标签的1值替换为1.0-label_smoothing，使得标签不会过于极端，即不会要么为1或要么为0
        target_onehot_labelsmoothing = torch.clamp(target_onehot.float(), min=self.label_smoothing/(input.size(1)-1), max=1.0-self.label_smoothing)
        # 对输入进行softmax操作并添加一个小的常数以防止log(0)
        input_softmax = F.softmax(input, dim=1) + 1e-7
        # 计算softmax后的输入的对数，torch.log()即为对数ln
        input_logsoftmax = torch.log(input_softmax)
        # 计算交叉熵损失，此处*为点积
        ce = -1 * input_logsoftmax * target_onehot_labelsmoothing
        # 计算Focal Loss
        fl = torch.pow((1 - input_softmax), self.gamma) * ce
        # 将Focal Loss乘以每个类别的权重，f1.sum(1)即为对每个样本的Focal Loss求和，按列对同一行每一列元素求和
        #target.long()是索引，即对应的类别，从而提取出每个类别的权重
        fl = fl.sum(1) * self.weight[target.long()]
        # 返回损失的平均值
        return fl.mean()

class RDropLoss(nn.Module):
    def __init__(self, loss, a=0.3):
        super(RDropLoss, self).__init__()
        self.loss = loss
        self.a = a

    def forward(self, input, target: torch.Tensor) -> torch.Tensor:
        if type(input) is list:
            input1, input2 = input
            main_loss = (self.loss(input1, target) + self.loss(input2, target)) * 0.5
            kl_loss = self.compute_kl_loss(input1, input2)
            return main_loss + self.a * kl_loss
        else:
            return self.loss(input, target)
    
    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    

class FocalMarginLoss(torch.nn.Module):
    def __init__(self, class_sample_counts=[2378,6426,4740,283,912,656,838,976,520,131,456,124],
                 label_smoothing:float = 0.0,
                 weight: torch.Tensor = None,
                 gamma:float =2.0):
        """
        初始化 BalancedFocalMarginLoss
        :param class_sample_counts: 每个类别的样本数量列表或Tensor
        :param gamma: 调节因子，减少易分类样本的权重
        :param label_smoothing: 标签平滑参数，但不使用标签平滑
        :param weight: 每个类别的权重
        """
        super(FocalMarginLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma
        self.class_counts = torch.tensor(class_sample_counts, dtype=torch.int)
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 BalancedFocalMarginLoss
        :param inputs: 模型输出，维度 [batch_size, num_classes]
        :param targets: 真实标签，维度 [batch_size]，每个值为类别索引
        """
        # 计算样本边界margin
        # 对输入进行softmax操作并添加一个小的常数以防止log(0)
        input_softmax = F.softmax(inputs, dim=1) + 1e-7
        device = 'cuda:0'  # 或者你想使用的其他设备
        input_softmax = input_softmax.to(device)
        targets = targets.to(device)
        arange_tensor = torch.arange(input_softmax.shape[1]).to(device)
        # 提取样本其他类别的最大预测概率
        max_other_class_probs, _ = input_softmax.masked_fill(arange_tensor.reshape(1, -1) == targets.reshape(-1, 1), value=0).max(dim=1)
        # 提取样本真实类别的预测概率
        true_class_probs = input_softmax[torch.arange(len(targets)), targets]
        # 计算样本边界margin，即真实类别的预测概率与其他类别的最大预测概率之差
        margin = true_class_probs - max_other_class_probs
        # 计算input_logsoftmax，取对数
        input_logsoftmax = torch.log(input_softmax)
        # 计算交叉熵损失
        ce = -1 * input_logsoftmax[torch.arange(len(targets)), targets]
        # 计算BalancedFocalMarginLoss
        bfm_loss = torch.pow((1 - margin), self.gamma) * ce
        # 返回损失的平均值
        return bfm_loss.mean()

class PSoftmaxLoss(nn.Module):
    def __init__(self, label_smoothing:float = 0.0, weight: torch.Tensor = None, gamma:float = 1.8):
        super(PSoftmaxLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 对目标标签进行one-hot编码
        target_onehot = F.one_hot(target, num_classes=input.size(1))
        # 对one-hot编码的标签应用标签平滑，将标签的0值替换为label_smoothing/(input.size(1)-1)，将标签的1值替换为1.0-label_smoothing，使得标签不会过于极端，即不会要么为1或要么为0
        target_onehot_labelsmoothing = torch.clamp(target_onehot.float(), min=self.label_smoothing/(input.size(1)-1), max=1.0-self.label_smoothing)
        # 对输入进行softmax操作并添加一个小的常数以防止log(0)
        input_softmax = F.softmax(input, dim=1) + 1e-7
        # 计算softmax后的输入的对数，torch.log()即为对数ln
        input_logsoftmax = torch.log(input_softmax)
        # 计算交叉熵损失，此处*为点积
        ce = -1 * input_logsoftmax * target_onehot_labelsmoothing
        # 计算PSoftmaxLos
        fl = (self.gamma/(1 + input_softmax* target_onehot_labelsmoothing)) * ce
        # 将PSoftmaxLos乘以每个类别的权重，f1.sum(1)即为对每个样本的PSoftmaxLos求和，按列对同一行每一列元素求和
        #target.long()是索引，即对应的类别，从而提取出每个类别的权重
        fl = fl.sum(1) * self.weight[target.long()]
        # 返回损失的平均值
        return fl.mean()



class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5,label_smoothing: float =0.0, weight: torch.Tensor = None):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-4
        self.mse = nn.MSELoss(reduction='sum')
        self.label_smoothing = label_smoothing

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        labels = F.one_hot(labels, num_classes=12)
        # 应用标签平滑
        labels = torch.clamp(labels.float(), min=self.label_smoothing/(logits.size(1)-1), max=1.0-self.label_smoothing)

        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)

        # Combine two losses
        loss = margin_loss + self.reconstruction_loss_scalar * reconstruction_loss
        return loss.mean()
