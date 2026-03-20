import torch
from torch import nn
from torch.nn import functional as F

__all__ = ['CapsNet', 'CapsuleLoss']

# Available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Fire(nn.Module):
    #声明Fire模块的超参数，构建Fire模块中的squeeze和expand操作，以RELU作为激活函数
    def __init__(
        self,
        inplanes: int, # 输入通道数
        squeeze_planes: int, #squeeze输出通道
        expand1x1_planes: int, # 1*1输出通道
        expand3x3_planes: int # 3*3输出通道
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    #构建Fire模块中的前向传播过程，通过cat将1x1和3x3融合成expand操作
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: out_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, out_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)

# Tanimoto相似度计算
def tanimoto_similarity(x, y):
    dot_product = (x * y).sum(dim=-1)
    return dot_product / (x.pow(2).sum(dim=-1) + y.pow(2).sum(dim=-1) - dot_product)

class DigitCaps(nn.Module):
    """Digit capsule layer."""

    def __init__(self, in_dim, in_caps, out_caps, out_dim, num_routing):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality of each capsule vector.
            in_caps: 		Number of input capsules if digits layer.
            out_caps: 		Number of capsules in the capsule layer
            out_dim: 		Dimensionality, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(DigitCaps, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing
        self.device = device
        self.W = nn.Parameter(0.01 * torch.randn(1, out_caps, in_caps, out_dim, in_dim),
                              requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        # W @ x =
        # (1, out_caps, in_caps, out_dim, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, out_caps, in_caps, out_dims, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, out_caps, in_caps, out_dim)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        b = torch.zeros(batch_size, self.out_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing - 1):
            # (batch_size, out_caps, in_caps, 1) -> Softmax along out_caps
            c = b.softmax(dim=1)

            # element-wise multiplication
            # (batch_size, out_caps, in_caps, 1) * (batch_size, in_caps, out_caps, out_dim) ->
            # (batch_size, out_caps, in_caps, out_dim) sum across in_caps ->
            # (batch_size, out_caps, out_dim)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along out_dim
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, out_caps, in_caps, out_dim) @ (batch_size, out_caps, out_dim, 1)
            # -> (batch_size, out_caps, in_caps, 1)

            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

            # Tanimoto相似度计算，不解决了，反正也没用

            # similarity = tanimoto_similarity(temp_u_hat, v.unsqueeze(-1))
            # b += similarity

        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along out_dim
        v = squash(s)

        return v


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        self.conv = nn.Conv2d(1, 256, 9)
        self.relu = nn.ReLU(inplace=True)

        # fire模块
        self.fire = Fire(256, 32, 128, 128)

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)

        # 定义dropout
        self.dropout = nn.Dropout(0.25)

        # Digit capsule
        self.digit_caps = DigitCaps(in_dim=8,
                                    in_caps=32 * 6 * 6,
                                    out_caps=12, # 12类
                                    out_dim=16,
                                    num_routing=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784), # 28*28
            nn.Sigmoid())

    def forward(self, x):
        out = self.relu(self.conv(x))
        # 插入fire模块
        out = self.fire(out)
        out = self.primary_caps(out)
        # 插入dropout
        out = self.dropout(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, out_capsules)
        logits = torch.norm(out, dim=-1)
        pred = torch.eye(12).to(device).index_select(dim=0, index=torch.argmax(logits, dim=1))

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * pred.unsqueeze(2)).contiguous().view(batch_size, -1))

        # 返回预测结果和重构结果 （需要重写训练框架代码，烦死了）
        return logits, reconstruction



# class CapsuleLoss(nn.Module):
#     def forward(self, data,target, output,  reconstructions):
#         # Margin loss
#         zero = torch.zeros(1)
#         margin_loss = target * torch.clamp(0.9 - output, min=0.) ** 2 \
#                     + 0.5 * (1. - target) * torch.clamp(output - 0.1, min=0.) ** 2
#         margin_loss = margin_loss.sum()

#         # Reconstruction loss
#         reconstruction_loss = F.mse_loss(reconstructions, data.view(reconstructions.size()[0], -1))

#         return (margin_loss + 0.0005 * reconstruction_loss)

class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.reconstruction_loss_scalar = 5e-4
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        # 打印left和labels的shape
        # print(labels.shape,left.shape)
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)
        print(reconstructions.shape,images.shape)
        # Reconstruction loss
        reconstruction_loss = self.mse(reconstructions.contiguous().view(images.shape), images)

        # Combine two losses
        loss = margin_loss + self.reconstruction_loss_scalar * reconstruction_loss
        return loss.mean()

if __name__ == "__main__":
    # 创建一个CapsNet实例
    model = CapsNet().to(device)

    # 生成一些随机输入数据
    # 假设我们有一个批次包含128个样本，每个样本是1x28x28的图像
    data = torch.randn(128, 1, 28, 28).to(device)

    # 将数据传递给模型
    logits, reconstruction = model(data)
    # print(logits.shape, reconstruction.shape)
    # 计算损失
    loss = CapsuleLoss()(data, torch.randn(128, 12).to(device), logits, reconstruction)
    # 打印损失
    print(loss)


