import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, torch, argparse, datetime, tqdm, random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from utils import utils_aug
from utils.utils import predict_single_image, cam_visual, dict_to_PrettyTable, select_device, model_fuse
from utils.utils_model import select_model
import pandas as pd
from io import StringIO
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_options(opt, **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            setattr(opt, key, value)
def parse_opt(source_in, save_path_in, cam_visual_in, cam_type_in, half_in):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r'', help='source data path(file, folder)')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label path')
    parser.add_argument('--save_path', type=str, default=r'runs/exp', help='save path for model and log')
    parser.add_argument('--test_tta', action="store_true", help='using TTA Tricks')
    parser.add_argument('--cam_visual', action="store_true", help='visual cam')
    parser.add_argument('--cam_type', type=str, choices=['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad'], default='FullGrad', help='cam type')
    parser.add_argument('--half', action="store_true", help='use FP16 half-precision inference')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_known_args()[0]
    
    set_options(opt,source = source_in, save_path = save_path_in,
                 cam_visual = cam_visual_in, cam_type = cam_type_in, half = half_in)
    # 如果cam_visual为true则device为cpu
    if opt.cam_visual:
        opt.device = 'cpu'
    if not os.path.exists(os.path.join(opt.save_path, 'best.pt')):
        raise Exception('best.pt not found. please check your --save_path folder')
    ckpt = torch.load(os.path.join(opt.save_path, 'best.pt'))
    DEVICE = select_device(opt.device)
    if opt.half and DEVICE.type == 'cpu':
        raise Exception('half inference only supported GPU.')
    if opt.half and opt.cam_visual:
        raise Exception('cam visual only supported cpu. please set device=cpu.')
    if (opt.device != 'cpu') and opt.cam_visual:
        raise Exception('cam visual only supported FP32.')

    with open(opt.label_path, encoding='utf-8') as f:
        CLASS_NUM = len(f.readlines())
    model = select_model(ckpt['model'].name, CLASS_NUM)
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    model_fuse(model)
    model = (model.half() if opt.half else model)
    model.to(DEVICE)
    model.eval()
    train_opt = ckpt['opt']
    set_seed(train_opt.random_seed)

    print('found checkpoint from {}, model type:{}\n{}'.format(opt.save_path, ckpt['model'].name, dict_to_PrettyTable(ckpt['best_metrice'], 'Best Metrice')))
    test_transform = utils_aug.get_dataprocessing_teststage(train_opt, opt, torch.load(os.path.join(opt.save_path, 'preprocess.transforms')))

    try:
        with open(opt.label_path, encoding='utf-8') as f:
            label = list(map(lambda x: x.strip(), f.readlines()))
    except Exception as e:
        with open(opt.label_path, encoding='gbk') as f:
            label = list(map(lambda x: x.strip(), f.readlines()))

    return opt, DEVICE, model, test_transform, label


def predict_main(source_in, save_path_in, cam_visual_in, cam_type_in, half_in):
    opt, DEVICE, model, test_transform, label = parse_opt(source_in, save_path_in, cam_visual_in, cam_type_in, half_in)

    if opt.cam_visual:
        cam_model = cam_visual(model, test_transform, DEVICE, opt)

    # 如果源路径是一个目录
    if os.path.isdir(opt.source):
        # 创建一个以当前时间为名字的新目录，用于保存预测结果
        save_path = os.path.join(opt.save_path, 'predict', datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
        os.makedirs(os.path.join(save_path))
        df_all = pd.DataFrame(columns=['图片路径', '预测类别', '预测类别概率'])
        result = []
        # 对目录中的每一张图像进行预测
        for file in tqdm.tqdm(os.listdir(opt.source)):
            # 对单张图像进行预测
            pred, pred_result = predict_single_image(os.path.join(opt.source, file), model, test_transform, DEVICE, half=opt.half)
            # 保存预测结果
            result.append('{},{},{}'.format(file, label[pred], pred_result[pred]))
            # 创建一个新的图像，并设置其大小为6x6
            plt.figure(figsize=(6, 6))
            # 如果启用了CAM可视化
            if opt.cam_visual:
                # 使用CAM模型对图像进行处理，并显示处理后的图像
                cam_output = cam_model(os.path.join(opt.source, file))
                plt.imshow(cam_output)
            else:
                # 显示原图
                plt.imshow(plt.imread(os.path.join(opt.source, file)))
            # 关闭坐标轴
            plt.axis('off')
            # 设置图像的标题，包括预测的标签和预测的概率
            plt.title('预测标签:{}\n预测概率:{:.4f}'.format(label[pred], float(pred_result[pred])))
            # 调整图像的布局
            plt.tight_layout()
            # 保存图像
            plt.savefig(os.path.join(save_path, file))
            # 清除当前图像
            plt.clf()
            # 关闭图像
            plt.close()

        # 将result转换为一个字符串，每个元素用换行符分隔
        result_str = '\n'.join(result)

        # 使用StringIO将字符串转换为文件对象，然后使用pandas的read_csv函数从文件对象创建DataFrame
        df_all = pd.read_csv(StringIO(result_str), header=None, names=['图片路径', '预测类别', '预测类别概率'])
        # 将所有的预测结果写入到一个xlsx文件中
        df_all.to_excel(os.path.join(save_path, '抽油机井工况诊断结果.xlsx'), index=False)
        # with open(os.path.join(save_path, '抽油机井工况诊断结果.xlsx'), 'w+') as f:
        #     f.write('图片路径,预测类别,预测类别概率\n')
        #     f.write('\n'.join(result))
        return save_path
    # 如果源路径是一个文件
    elif os.path.isfile(opt.source):
        # 创建一个以当前时间为名字的新目录，用于保存预测结果
        save_path = os.path.join(opt.save_path, 'predict', datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
        os.makedirs(os.path.join(save_path))
        # 对单个文件进行预测
        pred, pred_result = predict_single_image(opt.source, model, test_transform, DEVICE, half=opt.half)
        # 创建一个新的图像，并设置其大小为6x6
        plt.figure(figsize=(6, 6))
        # 如果启用了CAM可视化
        if opt.cam_visual:
            # 使用CAM模型对图像进行处理，并显示处理后的图像
            cam_output = cam_model(opt.source)
            plt.imshow(cam_output)
        else:
            # 显示原图
            plt.imshow(plt.imread(opt.source))
        # 关闭坐标轴
        plt.axis('off')
        # 设置图像的标题，包括预测的标签和预测的概率
        plt.title('预测标签:{}\n预测概率:{:.4f}'.format(label[pred], float(pred_result[pred])))
        # 调整图像的布局
        plt.tight_layout()
        # 保存图像
        plt.savefig(os.path.join(save_path, 'predict.png'))
        # 清除当前图像
        plt.clf()
        # 关闭图像
        plt.close()
        save_path = os.path.join(save_path, 'predict.png')
        return save_path,label,pred_result,pred




if __name__ == '__main__':
    opt, DEVICE, model, test_transform, label = parse_opt()

    if opt.cam_visual:
        cam_model = cam_visual(model, test_transform, DEVICE, opt)

    # 如果源路径是一个目录
    if os.path.isdir(opt.source):
        # 创建一个以当前时间为名字的新目录，用于保存预测结果
        save_path = os.path.join(opt.save_path, 'predict', datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S'))
        os.makedirs(os.path.join(save_path))
        result = []
        # 对目录中的每一张图像进行预测
        for file in tqdm.tqdm(os.listdir(opt.source)):
            # 对单张图像进行预测
            pred, pred_result = predict_single_image(os.path.join(opt.source, file), model, test_transform, DEVICE, half=opt.half)
            # 保存预测结果
            result.append('{},{},{}'.format(os.path.join(opt.source, file), label[pred], pred_result[pred]))
            
            # 创建一个新的图像，并设置其大小为6x6
            plt.figure(figsize=(6, 6))
            # 如果启用了CAM可视化
            if opt.cam_visual:
                # 使用CAM模型对图像进行处理，并显示处理后的图像
                cam_output = cam_model(os.path.join(opt.source, file))
                plt.imshow(cam_output)
            else:
                # 显示原图
                plt.imshow(plt.imread(os.path.join(opt.source, file)))
            # 关闭坐标轴
            plt.axis('off')
            # 设置图像的标题，包括预测的标签和预测的概率
            plt.title('预测标签:{}\n预测概率:{:.4f}'.format(label[pred], float(pred_result[pred])))
            # 调整图像的布局
            plt.tight_layout()
            # 保存图像
            plt.savefig(os.path.join(save_path, file))
            # 清除当前图像
            plt.clf()
            # 关闭图像
            plt.close()
        
        # 将所有的预测结果写入到一个CSV文件中
        with open(os.path.join(save_path, 'result.csv'), 'w+') as f:
            f.write('img_path,pred_class,pred_class_probability\n')
            f.write('\n'.join(result))
    # 如果源路径是一个文件
    elif os.path.isfile(opt.source):
        # 对单个文件进行预测
        pred, pred_result = predict_single_image(opt.source, model, test_transform, DEVICE, half=opt.half)
        
        # 创建一个新的图像，并设置其大小为6x6
        plt.figure(figsize=(6, 6))
        # 如果启用了CAM可视化
        if opt.cam_visual:
            # 使用CAM模型对图像进行处理，并显示处理后的图像
            cam_output = cam_model(opt.source, pred)
            plt.imshow(cam_output)
        else:
            # 显示原图
            plt.imshow(plt.imread(opt.source))
        # 关闭坐标轴
        plt.axis('off')
        # 设置图像的标题，包括预测的标签和预测的概率
        plt.title('预测标签:{}\n预测概率:{:.4f}'.format(label[pred], float(pred_result[pred])))
        # 调整图像的布局
        plt.tight_layout()
        # 显示图像
        plt.show()