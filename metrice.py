import warnings, sys, datetime, random
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, torch, argparse, time, torchvision, tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from utils import utils_aug
from utils.utils import classification_metrice, Metrice_Dataset, visual_predictions, visual_tsne, dict_to_PrettyTable, Model_Inference, select_device, model_fuse,multiclass_roc_auc

torch.backends.cudnn.deterministic = True
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def set_options(opt, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            setattr(opt, key, value)
def parse_opt(save_path_in,task_in,batch_size_in,tsne_in,half_in):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=r'dataset/train', help='train data path')
    parser.add_argument('--val_path', type=str, default=r'dataset/val', help='val data path')
    parser.add_argument('--test_path', type=str, default=r'dataset/test', help='test data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--task', type=str, choices=['train', 'val', 'test', 'fps'], default='val', help='train, val, test, fps')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--test_tta', action="store_true", help='using TTA Tricks')
    parser.add_argument('--visual', action="store_true", help='visual dataset identification')
    parser.add_argument('--tsne', action="store_true", help='visual tsne')
    parser.add_argument('--half', action="store_true", help='use FP16 half-precision inference')
    parser.add_argument('--model_type', type=str, choices=['torch', 'torchscript', 'onnx', 'tensorrt'], default='torch', help='model type(default: torch)')

    opt = parser.parse_known_args()[0]

    set_options(opt, save_path=save_path_in, task=task_in, 
                batch_size=int(batch_size_in), tsne=tsne_in, half=half_in)
    DEVICE = select_device(opt.device, opt.batch_size)
    if opt.half and DEVICE.type == 'cpu':
        raise Exception('half inference only supported GPU.')
    if not os.path.exists(os.path.join(opt.save_path, 'best.pt')):
        raise Exception('best.pt not found. please check your --save_path folder')
    ckpt = torch.load(os.path.join(opt.save_path, 'best.pt'))
    train_opt = ckpt['opt']
    set_seed(train_opt.random_seed)
    model = Model_Inference(DEVICE, opt)

    print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))

    test_transform = utils_aug.get_dataprocessing_teststage(train_opt, opt, torch.load(os.path.join(opt.save_path, 'preprocess.transforms')))

    # 如果任务是计算每秒帧数（fps）
    if opt.task == 'fps':
        # 创建一个随机的输入张量，形状为 (batch_size, image_channel, image_size, image_size)，并将其移动到设备（DEVICE）上
        inputs = torch.rand((opt.batch_size, train_opt.image_channel, train_opt.image_size, train_opt.image_size)).to(DEVICE)
        # 如果启用了半精度浮点数（half）并且有可用的 CUDA 设备，将输入张量转换为半精度浮点数
        if opt.half and torch.cuda.is_available():
            inputs = inputs.half()
            torch.cuda.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # 设置预热次数和测试次数
        warm_up, test_time = 100, 300
        # 初始化 fps 数组
        fps_arr = []
        # 对预热次数和测试次数总和进行循环
        for i in tqdm.tqdm(range(test_time + warm_up)):
            # 记录开始时间
            since = time.perf_counter()
            # 在推理模式下运行模型
            with torch.inference_mode():
                model(inputs)
            # 如果当前循环次数大于预热次数，将运行模型所用的时间添加到 fps 数组
            if i > warm_up:
                torch.cuda.synchronize()
                fps_arr.append(time.perf_counter() - since)
        # 计算 fps 数组的平均值，得到每帧所需的平均时间（fps）
        fps = np.mean(fps_arr)
        # 打印每帧所需的平均时间（秒）、每秒帧数（fps）和批量大小，每帧就是每张图片！
        print('{:.6f} seconds, {:.2f} fps, @batch_size {}'.format(fps, 1 / fps, opt.batch_size))
        # 退出程序
        sys.exit(0)
    else:
        save_path = os.path.join(opt.save_path, opt.task, datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        CLASS_NUM = len(os.listdir(eval('opt.{}_path'.format(opt.task))))
        test_dataset = Metrice_Dataset(torchvision.datasets.ImageFolder(eval('opt.{}_path'.format(opt.task)), transform=test_transform))
        test_dataset = torch.utils.data.DataLoader(test_dataset, opt.batch_size, shuffle=False,
                                                   num_workers=(0 if opt.test_tta else opt.workers))

        try:
            with open(opt.label_path, encoding='utf-8') as f:
                label = list(map(lambda x: x.strip(), f.readlines()))
        except Exception as e:
            with open(opt.label_path, encoding='gbk') as f:
                label = list(map(lambda x: x.strip(), f.readlines()))

        return opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path,train_opt


def measure_fps(model, opt, train_opt, DEVICE):
    # 创建一个随机的输入张量，形状为 (batch_size, image_channel, image_size, image_size)，并将其移动到设备（DEVICE）上
    inputs = torch.rand((opt.batch_size, train_opt.image_channel, train_opt.image_size, train_opt.image_size)).to(DEVICE)
    # 如果启用了半精度浮点数（half）并且有可用的 CUDA 设备，将输入张量转换为半精度浮点数
    if opt.half and torch.cuda.is_available():
        inputs = inputs.half()
        torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # 设置预热次数和测试次数
    warm_up, test_time = 100, 300
    # 初始化 fps 数组
    fps_arr = []
    # 对预热次数和测试次数总和进行循环
    for i in tqdm.tqdm(range(test_time + warm_up)):
        # 记录开始时间
        since = time.perf_counter()
        # 在推理模式下运行模型
        with torch.inference_mode():
            model(inputs)
        # 如果当前循环次数大于预热次数，将运行模型所用的时间添加到 fps 数组
        if i > warm_up:
            torch.cuda.synchronize()
            fps_arr.append(time.perf_counter() - since)
    # 计算 fps 数组的平均值，得到每帧所需的平均时间（fps）
    fps = np.mean(fps_arr)
    # 打印每帧所需的平均时间（秒）、每秒帧数（fps）和批量大小，每帧就是每张图片！
    print('{:.6f} seconds, {:.2f} fps, @batch_size {}'.format(fps, 1 / fps, opt.batch_size))
    return fps,1/fps
 
# 用于web的界面函数
def metrice_main(save_path_in,task_in,batch_size_in,tsne_in,half_in):
    opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path,train_opt = parse_opt(save_path_in,task_in,batch_size_in,tsne_in,half_in)
    y_true, y_pred, y_score, y_feature, img_path = [], [], [], [], []
    # 固定随机种子
    set_seed(0)

    # 进行推理时间计算
    inference_time,fps=measure_fps(model, opt, train_opt, DEVICE)

    with torch.inference_mode():
        for x, y, path in tqdm.tqdm(test_dataset, desc='Test Stage'):
            x = (x.half().to(DEVICE) if opt.half else x.to(DEVICE))
            if opt.test_tta:
                bs, ncrops, c, h, w = x.size()
                pred = model(x.view(-1, c, h, w))
                pred = pred.view(bs, ncrops, -1).mean(1)

                if opt.tsne:
                    pred_feature = model.forward_features(x.view(-1, c, h, w))
                    pred_feature = pred_feature.view(bs, ncrops, -1).mean(1)
            else:
                pred = model(x)

                if opt.tsne:
                    # 使用模型的 forward_features 方法对输入 x 进行前向传播，获取预测特征
                    pred_feature = model.forward_features(x)
            try:
                pred = torch.softmax(pred, 1)
            except:
                pred = torch.softmax(torch.from_numpy(pred), 1) # using torch.softmax will faster than numpy

            y_true.extend(list(y.cpu().detach().numpy()))
            y_pred.extend(list(pred.argmax(-1).cpu().detach().numpy()))
            y_score.extend(list(pred.max(-1)[0].cpu().detach().numpy()))
            img_path.extend(list(path))

            # 如果启用了 t-SNE（一种用于高维数据可视化的工具），则将预测特征从 GPU 移动到 CPU，并将其从 PyTorch 张量转换为 numpy 数组，然后添加到 y_feature 列表中
            if opt.tsne:
                y_feature.extend(list(pred_feature.cpu().detach().numpy()))

    classification_metrice(np.array(y_true), np.array(y_pred), CLASS_NUM, label, save_path,inference_time,fps)

    if opt.visual:
        visual_predictions(np.array(y_true), np.array(y_pred), np.array(y_score), np.array(img_path), label, save_path)
    if opt.tsne:
        visual_tsne(np.array(y_feature), np.array(y_pred), np.array(img_path), label, save_path)
    
    return save_path

if __name__ == '__main__':
    opt, model, test_dataset, DEVICE, CLASS_NUM, label, save_path,train_opt = parse_opt()
    y_true, y_pred, y_score, y_feature, img_path = [], [], [], [], []
    # 固定随机种子
    set_seed(0)

    # 进行推理时间计算
    inference_time,fps=measure_fps(model, opt, train_opt, DEVICE)

    with torch.inference_mode():
        for x, y, path in tqdm.tqdm(test_dataset, desc='Test Stage'):
            x = (x.half().to(DEVICE) if opt.half else x.to(DEVICE))
            if opt.test_tta:
                bs, ncrops, c, h, w = x.size()
                pred = model(x.view(-1, c, h, w))
                pred = pred.view(bs, ncrops, -1).mean(1)

                if opt.tsne:
                    pred_feature = model.forward_features(x.view(-1, c, h, w))
                    pred_feature = pred_feature.view(bs, ncrops, -1).mean(1)
            else:
                # pred = model(x)
                if 'fdt_capsnet' in opt.save_path:
                    pred, _ = model(x)
                else:
                    pred = model(x)

                if opt.tsne:
                    # 使用模型的 forward_features 方法对输入 x 进行前向传播，获取预测特征
                    pred_feature = model.forward_features(x)
            try:
                pred = torch.softmax(pred, 1)
            except:
                pred = torch.softmax(torch.from_numpy(pred), 1) # using torch.softmax will faster than numpy

            y_true.extend(list(y.cpu().detach().numpy()))
            y_pred.extend(list(pred.argmax(-1).cpu().detach().numpy()))
            y_score.extend(list(pred.max(-1)[0].cpu().detach().numpy()))
            img_path.extend(list(path))

            # 如果启用了 t-SNE（一种用于高维数据可视化的工具），则将预测特征从 GPU 移动到 CPU，并将其从 PyTorch 张量转换为 numpy 数组，然后添加到 y_feature 列表中
            if opt.tsne:
                y_feature.extend(list(pred_feature.cpu().detach().numpy()))

    classification_metrice(np.array(y_true), np.array(y_pred), CLASS_NUM, label, save_path,inference_time,fps)


    # # 绘制roc曲线,直接放到classification_metrice函数里面了
    # multiclass_roc_auc( np.array(y_true), np.array(y_pred),label,save_path)



    if opt.visual:
        visual_predictions(np.array(y_true), np.array(y_pred), np.array(y_score), np.array(img_path), label, save_path)
    if opt.tsne:
        visual_tsne(np.array(y_feature), np.array(y_pred), np.array(img_path), label, save_path)