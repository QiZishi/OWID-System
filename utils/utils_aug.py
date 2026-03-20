import torch, tqdm
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from copy import deepcopy
import albumentations as A
import random, math
from mainweb_globalValue import stop_training_flag
def get_mean_and_std(dataset, opt):
    '''计算数据集的均值和标准差'''
    if opt.imagenet_meanstd:
        if opt.image_channel == 1:
            print('使用ImageNet的均值和标准差。均值：[0.485] 标准差：[0.229]。')
            return [0.485], [0.229]
        else:
            print('使用ImageNet的均值和标准差。均值：[0.485, 0.456, 0.406] 标准差：[0.229, 0.224, 0.225]。')
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        print('计算数据集的均值和方差...')
        mean = torch.zeros(opt.image_channel)
        std = torch.zeros(opt.image_channel)
        for inputs, targets in tqdm.tqdm(dataset):
            inputs = transforms.ToTensor()(inputs)
            for i in range(opt.image_channel):
                mean[i] += inputs[i, :, :].mean()
                std[i] += inputs[i, :, :].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        print('计算完成。均值：{} 标准差：{}.'.format(list(mean.detach().numpy()), list(std.detach().numpy())))
        return mean, std

# 最大最小归一化
def min_max_norm(image):
    min_value = torch.min(image)
    max_value = torch.max(image)
    norm_image = (image - min_value) / (max_value - min_value)
    return norm_image
def get_processing(dataset, opt):
    # 创建一个转换序列，首先将输入图像转换为张量，然后使用数据集的均值和标准差对图像进行归一化
    # return transforms.Compose([transforms.ToTensor()])
    # 先归一化再标准化
    return transforms.Compose(
        [
        transforms.ToTensor(),  # 将输入图像转换为张量，归一化
        transforms.Normalize(*get_mean_and_std(dataset, opt)) # 使用数据集的均值和标准差对图像进行标准化
        ]) # 最大最小归一化 不小心删了。。。



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, opt, alpha=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    if opt.mixup == 'mixup':
        mixed_x = lam * x + (1 - lam) * x[index, :]
    elif opt.mixup == 'cutmix':
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        mixed_x = deepcopy(x)
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    else:
        raise 'Unsupported MixUp Methods.'
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def select_Augment(opt):
    if opt.Augment == 'RandAugment':
        return transforms.RandAugment()
    elif opt.Augment == 'AutoAugment':
        return transforms.AutoAugment()
    elif opt.Augment == 'TrivialAugmentWide':
        return transforms.TrivialAugmentWide()
    elif opt.Augment == 'AugMix':
        return transforms.AugMix()
    elif opt.Augment == 'CutOut':
        return CutOut()
    elif opt.Augment == 'RandomErasing':
        return RandomErasing()
    else:
        return None
# 建立一个二值化函数
def binaryzation(img):
    img = img.convert('L')
    pixdata = img.load()
    w, h = img.size
    threshold = 127
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img
# 调用二值化函数
def binaryzation_func(img):
    return binaryzation(img)


def get_dataprocessing(dataset, opt, preprocess=None):
    # 获取数据处理的方法
    if not preprocess:  # 如果预处理方法为空
        preprocess = get_processing(dataset, opt)  # 获取预处理方法
        torch.save(preprocess, r'{}/preprocess.transforms'.format(opt.save_path))  # 保存预处理方法
        
    if len(opt.custom_augment.transforms) == 0:  # 如果自定义增强方法为空
        augment = select_Augment(opt)  # 选择增强方法
    else:
        augment = opt.custom_augment  # 使用自定义的增强方法

    if augment is None:  # 如果增强方法为空
        '''由于是为了放大照片，所以对transforms.Resize的源码进行了修改'''
        '''使用了三次样条插值BICUBIC，而不是默认的双线性插值BILINEAR'''
        train_transform = transforms.Compose(
            [
                # # 将图片转换为灰度图
                # transforms.Grayscale(),
                # # 对图片进行二值化
                # transforms.Lambda(binaryzation_func),
                # # 将图片resize到320*240
                # transforms.Resize(((320,240))),

                # 随机平移、缩放、小角度旋转图像,degrees=0表示不旋转,degrees=(-2,2)表示随机旋转-2到2度，scale=（0.5，1.5）表示图像大小在原始图50%到150%之间，translate=(0.2, 0.2)表示在水平和垂直方向上平移的最大距离
                transforms.RandomAffine(degrees=(-2,2), translate=(0.1, 0.1), scale=(0.5, 1.5), fill=(255,255,255)),
                # 随机水平翻转图像
                transforms.RandomHorizontalFlip(p=0.5),

                # transforms.Resize((int(opt.image_size+ opt.image_size*0.1) )),  # 调整图片大小
                # transforms.RandomCrop((opt.image_size, opt.image_size)),  # 随机裁剪图片
                # transforms.Resize((opt.image_size,opt.image_size)), # 重构图片大小
                preprocess  # 应用预处理方法
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),  # 调整图片大小
                transforms.RandomCrop((opt.image_size, opt.image_size)),  # 随机裁剪图片
                augment,  # 应用增强方法
                preprocess  # 应用预处理方法
            ]
        )

    if opt.test_tta:  # 如果测试时使用测试时间增强
        test_transform = transforms.Compose([
            transforms.Resize((int(opt.image_size + opt.image_size * 0.1))),  # 调整图片大小
            transforms.TenCrop((opt.image_size, opt.image_size)),  # 将图片分为10个部分
            transforms.Lambda(lambda crops: torch.stack([preprocess(crop) for crop in crops])),  # 对每个部分应用预处理方法并拼接
        ])
    else:
        test_transform = transforms.Compose([
                # # 将图片转换为灰度图
                # transforms.Grayscale(),
                # # 对图片进行二值化
                # transforms.Lambda(binaryzation_func),
                # # 将图片resize到320*240
                # transforms.Resize(((320,240))),
            # 直接重构为224*224，不再中心裁剪
            # transforms.Resize((opt.image_size,opt.image_size)),  # 调整图片大小
            # transforms.CenterCrop((opt.image_size, opt.image_size)),  # 中心裁剪图片
            preprocess  # 应用预处理方法
        ])

    return train_transform, test_transform  # 返回训练和测试时的转换方法

# 测试阶段使用的数据处理方法
def get_dataprocessing_teststage(train_opt, opt, preprocess):
    if opt.test_tta:
        test_transform = transforms.Compose([
            transforms.Resize((int(train_opt.image_size + train_opt.image_size * 0.1))),
            transforms.TenCrop((train_opt.image_size, train_opt.image_size)),
            transforms.Lambda(lambda crops: torch.stack([preprocess(crop) for crop in crops]))
        ])
    else:
        test_transform = transforms.Compose([
                # # 将图片转换为灰度图
                # transforms.Grayscale(),
                # # 对图片进行二值化
                # transforms.Lambda(binaryzation_func),
                # # 将图片resize到320*240
                # transforms.Resize(((320,240))),
            # transforms.Pad((0, 0, 0, train_opt.image_size - 112), fill=(255, 255, 255), padding_mode='constant'),
            # 直接重构为224*224，不再中心裁剪
            # transforms.Resize((train_opt.image_size,train_opt.image_size)),
            # transforms.CenterCrop((train_opt.image_size, train_opt.image_size)),
            preprocess
        ])
    return test_transform

class CutOut(object):
    def __init__(self, n_holes=4, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        mask = np.ones_like(img, np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
        return Image.fromarray(np.array(img * mask, dtype=np.uint8))
    def __str__(self):
        return 'CutOut'


class RandomErasing(object):
    def __init__(self, EPSILON = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    # img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img
    def __str__(self):
        return 'RandomErasing'


class Create_Albumentations_From_Name(object):
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/
    def __init__(self, name, **kwargs):
        self.name = name
        self.transform = eval('A.{}'.format(name))(**kwargs)

    def __call__(self, img):
        img = np.array(img)
        return Image.fromarray(np.array(self.transform(image=img)['image'], dtype=np.uint8))
    
    def __str__(self):
        return self.name