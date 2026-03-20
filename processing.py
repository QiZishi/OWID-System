import warnings
warnings.filterwarnings("ignore")
import os, shutil, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import random
from PIL import Image
from tqdm import tqdm
# set random seed
np.random.seed(0)
random.seed(0)
'''
    This file help us to split the dataset.
    It's going to be a training set, a validation set, a test set.
    We need to get all the image data into --data_path
    Example:
        dataset/train/dog/*.(jpg, png, bmp, ...)
        dataset/train/cat/*.(jpg, png, bmp, ...)
        dataset/train/person/*.(jpg, png, bmp, ...)
        and so on...
    
    program flow:
    1. generate label.txt.
    2. rename --data_path.
    3. split dataset.
'''

def processing_parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'dataset/train', help='all data path')
    parser.add_argument('--label_path', type=str, default=r'dataset/label.txt', help='label txt save path')
    parser.add_argument('--val_size', type=float, default=0.2, help='size of val set')
    parser.add_argument('--test_size', type=float, default=0
                        , help='size of test set')
    parser.add_argument('--pattern', type=str,choices=['mode1','mode2','mode3'], default='mode1', help='mode1为生产参数归一化,mode2为普通归一化,mode3为不归一化')
    parser.add_argument('--image_size1', type=int, default=224, help='图片宽度')
    parser.add_argument('--image_size2', type=int, default=112, help='图片高度')
    opt = parser.parse_known_args()[0]
    return opt


# 下面是我自己写的函数
def process_image(folder_path_new, folder, file): # 已弃用
    # 图像处理函数：将原始图像转换为灰度图像并进行二值化处理
    '''
    folder_path_new: 图像所在文件夹
    folder: 图像所在文件夹的下一级文件夹
    file: 图像名称
    image_size1:图像宽度
    image_size2:图像高度
    '''
    # 使用imread重新读取数据
    img = cv2.imread(folder_path_new+folder+'/'+file[:-4]+'.png',cv2.IMREAD_UNCHANGED)
    # 将图片转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用OpenCV的adaptiveThreshold函数对灰度图像进行二值化处理
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # 将生成的图片保存到文件夹中
    cv2.imwrite(folder_path_new+folder+'/'+file[:-4]+'.png',thresh)

def zero_data(x, f):
    # case 1 : zero
    case1 = (x == 0).all() | (f == 0).all()
    # case 2 : close to zero
    case2 = abs(max(f) - min(f)) < 1
    return case1 or case2


# 绘制示功图（蓝色曲线为上半程，红色曲线为下半程，绿色闭合曲线为全程）
def plot_and_save(df, folder_path_new, folder, file,image_size1,image_size2):
    # image_size1:图像宽度
    # image_size2:图像高度

    # 设置线宽为2
    plt.rcParams['lines.linewidth'] = 2
    # 数据点1至位移最大的那个点为蓝色曲线
    plt.plot(df['位移(m)'][1:df['位移(m)'].idxmax()+1], df['载荷(kn)'][1:df['位移(m)'].idxmax()+1], color='blue')
    # 数据点位移最大的那个点至最后为红色曲线
    plt.plot(df['位移(m)'][df['位移(m)'].idxmax()+1:], df['载荷(kn)'][df['位移(m)'].idxmax()+1:], color='red')
    # 绘制整个闭合曲线，颜色为绿色，线宽为2，用绿色填充
    plt.fill(df['位移(m)'], df['载荷(kn)'], color='green')

    # # 设置y轴范围为0到最大值+（最大值-最小值）*0.1
    # plt.ylim(0, df['载荷(kn)'].max()+(df['载荷(kn)'].max()-df['载荷(kn)'].min())*0.1)
    # # 设置x轴范围为最小值-0.1到最大值+0.1
    # plt.xlim(df['位移(m)'].min()-0.1, df['位移(m)'].max()+0.1)

    # 去除坐标
    plt.xticks([])
    plt.yticks([])
    # 去除边框
    plt.axis('off')
    # 去除坐标框
    plt.box(False)
    # 设置图像分辨率
    plt.gcf().set_size_inches(image_size1/100,image_size2/100)
    # 图片背景设置为白色
    plt.gcf().set_facecolor('white')
    # bbox_inches='tight'表示去除白边(不使用了)
    plt.savefig(folder_path_new+folder+'/'+file+'.png',format='png')
    # 擦除图像
    plt.clf()

# 生产参数归一化
def normalization_each_device(x, f, f_max, f_min):
    length = len(f)
    x_max = max(x)
    x_min = min(x)
    for i in range(length):
        x[i] = (x[i] - x_min) / (x_max - x_min)
        f[i] = (f[i] - f_min) / (f_max - f_min)
    return x, f
def process_data(origin_data_path, new_data_path, image_size1,image_size2,pattern):
    # 原始数据归一化
    scaler = MinMaxScaler()
    # 依次读取folder_path_new1文件夹中的文件夹
    for folder in os.listdir(os.path.join(origin_data_path)):
        # 如果文件夹不存在，则新建文件夹
        os.makedirs(os.path.join(new_data_path, folder), exist_ok=True)
        # 依次读取A01-A12文件夹中的csv文件
        for file in tqdm([f for f in os.listdir(os.path.join(origin_data_path,folder)) if f.endswith('.csv')]):
            # 读取文件路径
            file_path = os.path.join(origin_data_path, folder,file)
            # 读取文件
            df = pd.read_csv(file_path,encoding='gbk')
            f_max = max(df['载荷(kn)'].values)
            f_min = min(df['载荷(kn)'].values)
            num = int(file.split('_')[-2])
            sample = int(file.split('_')[-1].split('.')[0])
            for i in range(num):
                x = df['位移(m)'].values[i * sample:(i + 1) * sample]
                f = df['载荷(kn)'].values[i * sample:(i + 1) * sample]
                if pattern == 'mode1' and 'add' not in file:
                    # 生产设备参数归一化（载荷取表格中的最大和最小，位移取自己的最大和最小）
                    x_new, f_new = normalization_each_device(x, f, f_max, f_min)
                elif pattern == 'mode2' and  'add' not in file:
                    # 普通归一化
                    x_new = scaler.fit_transform(x.reshape(-1, 1)).reshape(-1)
                    f_new = scaler.fit_transform(f.reshape(-1, 1)).reshape(-1)
                else:
                    # 不归一化
                    x_new = x
                    f_new = f
                # 将x_new和f_new合并生成一个dataframe
                data = pd.DataFrame({'位移(m)': x_new, '载荷(kn)': f_new})
                file_name_new = '_'.join(file.split('_')[:-2]) + '_' + str(i)
                # file_name_new = '_'.join(file.split('_')[:-2]) + '_' + str(i) + 'L' + str(sample)
                # 画图并保存
                plot_and_save(data,new_data_path,folder, file_name_new,image_size1,image_size2)
                # 读取图片
                img_path = os.path.join(new_data_path, folder, file_name_new+'.png')
                img = Image.open(img_path)
                # 使用padding对图片进行填充，将原先为224*112的图片填充至224*224，填充颜色是白色
                img = transforms.Pad((0, 56), fill=255)(img)
                # 将图片保存
                img.save(img_path)



# 旋转图片代码
def rotate_image(image_path, angle,flag=True):
    # 读取图片
    img = Image.open(image_path)
    # 对图片进行angle度旋转
    img = img.rotate(angle)
    # 如果flag为True，则进行r通道和b通道的交换
    if flag:
        # 将r通道的值赋值给b通道，将b通道的值赋值给r通道
        r, g, b,_ = img.split()
        img = Image.merge("RGB", (b, g, r))
    return img

# 数据扩充函数
def augment_images(folder_path):
    a2dires = ['A0201', 'A0202', 'A0203', 'A0204', 'A0208', 'A0209', 'A0210', 'A0211'] #选择部分
    a11dires = ['A11_145819', 'B0101'] #选择部分
    for folder in os.listdir(folder_path):
        # A01 工作正常 --旋转180°-->  A01 工作正常
        if folder == '00':
            # 依次读取A01文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma01_'+file
                image_new_path = os.path.join(folder_path, folder, file_name)
                # 保存图片
                img.save(image_new_path)
        # 部分A02 供液不足 --旋转180--> A07 游动阀关闭迟缓
        elif folder == '01':
            # 依次读取A02文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                if file.split('_')[0] not in a2dires:
                    continue
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma02_'+file
                folder_new = '06'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)
        # A05 上碰泵 <--旋转180°--> A06 下碰泵
        elif folder == '04':
            # 依次读取A05文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma05_'+file
                folder_new = '05'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)
        # A06 下碰泵 <--旋转180°--> A05 上碰泵
        elif folder == '05':
            # 依次读取A06文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                if 'augfroma05' in file:
                    continue
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma06_'+file
                folder_new = '04'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)
        #  A0701 游动阀关闭迟缓--旋转180--> A02 供液不足
        #  A07_141987 游动阀关闭迟缓--旋转180--> A11 砂影响+供液不足
        elif folder == '06':
            # 依次读取A07文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                if  '_'.join(file.split('_')[:-1]) not in ['A0701' ,'A07_141987'] :
                    continue
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma07_'+file
                if '_'.join(file.split('_')[:-1]) == 'A0701':
                    folder_new = '01'
                else:
                    folder_new = '10'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)
        # A09 游动阀漏 <--旋转180°--> A10 固定阀漏
        elif folder == '08':
            # 依次读取A09文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma09_'+file
                folder_new = '09'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)
        # A10 固定阀漏 <--旋转180°--> A09 游动阀漏
        elif folder == '09':
            # 依次读取A10文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                if 'augfroma09' in file:
                    continue
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma10_'+file
                folder_new = '08'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)
        # 部分A11 砂影响+供液不足 --旋转180--> A07 游动阀关闭迟缓
        elif folder == '10':
            # 依次读取A11文件夹中的文件
            for file in os.listdir(os.path.join(folder_path, folder)):
                if '_'.join(file.split('_')[:-1]) not in a11dires :
                    continue
                # 读取图片路径
                image_path = os.path.join(folder_path, folder, file)
                # 旋转图片
                img = rotate_image(image_path, 180)
                file_name = 'augfroma11_'+file
                folder_new = '06'
                image_new_path = os.path.join(folder_path, folder_new, file_name)
                # 保存图片
                img.save(image_new_path)

 
    # 数据扩充完毕
        print('数据扩充完毕！')


def count_images_and_save(path, filepath,label): # 统计各个类别的图像数量
    # 检查路径是否存在
    if not os.path.exists(path):
        print(f"路径 '{path}' 不存在。")
        return
    # 文件总数
    total_count = 0
    print('统计{}文件夹中的各个类别的图像数量'.format(path))
    # 获取path路径下的所有子文件夹
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    with open(filepath, 'a+') as file:
        file.write('统计{}文件夹中的各个类别的图像数量\n'.format(path))
        for folder in subfolders:
            # 获取每个子文件夹的文件数量
            files_count = len(os.listdir(folder))
            # 累加文件数量
            total_count += files_count
            # 打印文件数量
            print(f"类别 '{label[int(os.path.basename(folder))]}' 中的文件数量为: {files_count}")
            # 将信息写入文件
            file.write(f"类别 '{label[int(os.path.basename(folder))]}' 中的文件数量为: {files_count}\n")
        # 打印总文件数量
        print(f"总文件数量为: {total_count}")
        # 将信息写入文件
        file.write(f"总文件数量为: {total_count}\n")

def move_files(origin_data_path, new_data_path1):
    for folder in os.listdir(origin_data_path):
        os.makedirs(os.path.join(new_data_path1, folder), exist_ok=True)
        # 依次读取A01-A12文件夹中的csv文件
        for file in [f for f in os.listdir(os.path.join(origin_data_path, folder)) if f.endswith('.csv')]:
            # 将csv文件复制到新文件夹中
            shutil.copy(os.path.join(origin_data_path, folder, file), os.path.join(new_data_path1, folder, file))

# 数据集划分函数（按照小类别划分）
def split_dataset(val_size, test_size,label):
    if not os.path.exists('dataset/val/'):
        os.makedirs('dataset/val/')
    if not os.path.exists('dataset/test/'):
        os.makedirs('dataset/test/')
    if not os.path.exists('dataset/other/'):
        os.makedirs('dataset/other/')
    # 如果label为空
    if not label:
        oil_class = ['00','01','02','03','04','05','06','07','08','09','10','11']
    else:
        # 读取label的长度，将其转化为'00'、‘01’、‘02’...的列表
        str_len = len(str(len(label)))  # 计算需要填充的零的数量
        oil_class = [str(i).zfill(str_len) for i in range(len(label))]  # 生成新的列表
    for index in oil_class:
        if not os.path.exists('dataset/val/{}'.format(index)):
            os.makedirs('dataset/val/{}'.format(index))
        if not os.path.exists('dataset/test/{}'.format(index)):
            os.makedirs('dataset/test/{}'.format(index))
        if not os.path.exists('dataset/other/{}'.format(index)):
            os.makedirs('dataset/other/{}'.format(index))
        train_val = pd.DataFrame()
        filename = []
        label = []
        bj = []
        # 新建列表jl_test用于存储测试集中的bj值
        jl_test = []
        for i in os.listdir("dataset/train/{}".format(index)):
            if ".png" in i:
                filename.append(i)
                label.append(index)
                bj.append("_".join(i.split("_")[:-1]))
        train_val['filename'] = filename
        train_val['label']= label
        train_val['bj'] = bj
        # 获取bj列的唯一值，并打乱顺序
        '''jl是诸如A0106这种格式元素的列表，用于划分数据集，相当于将A1大类进行进一步划分几个小类，
        个人认为A0106表示型号为06的抽油机井的A01工况表现，应该是尽可能将一种型号的抽油机井的工况表现划分到一起，
        这样可以减少训练集和验证集中的数据重复性，增加模型的泛化能力'''
        jl = train_val['bj'].unique()
        tot = train_val.shape[0]
        print('总样本数为：',tot)
        # # 以“test”为关键词对jl列表进行排序，将含有“test”的元素放在列表的最后
        # jl = sorted(jl, key=lambda x: ('test' in x, train_val[train_val['bj']==x].shape[0]), reverse=False)
        # # 从小到大排序
        # # jl = sorted(jl,key=lambda x:train_val[train_val['bj']==x].shape[0],reverse= False)
        # # 从大到小排序
        # # jl = sorted(jl,key=lambda x:train_val[train_val['bj']==x].shape[0],reverse= True)
        # print(jl)
        # '''在迭代过程中不能修改jl,所以复制一份！'''
        # jl_copy = jl.copy()
        # sm = 0
        # for j in jl_copy:
        #     deal = train_val[train_val['bj']==j]
        #     if sm < round(tot * (1-val_size-test_size)) :
        #         sm = sm+deal.shape[0]
        #         continue
        #     else:
        #         for i in range(deal.shape[0]):
        #             shutil.move('dataset/train/{}/'.format(index)+deal['filename'].values[i],'dataset/val/{}/'.format(index))
        #         # 将j添加到jl_test列表中
        #         jl_test.append(j)
        #         # 将j从jl列表中删除
        #         jl.remove(j)
        # 以“test”为关键词对jl列表进行排序，将含有“test”的元素放在列表的最前
        jl = sorted(jl, key=lambda x: ('test' in x, -train_val[train_val['bj']==x].shape[0]), reverse=True)
        # 从小到大排序
        # jl = sorted(jl,key=lambda x:train_val[train_val['bj']==x].shape[0],reverse=False)
        # 从大到小排序
        # jl = sorted(jl,key=lambda x:train_val[train_val['bj']==x].shape[0],reverse= True)
        # print(jl)
        split_flag = False
        val_count = len(os.listdir('dataset/val/{}/'.format(index)))
        jl_copy = jl.copy()
        if val_count < round(tot *(val_size+test_size)) :
            for j in jl_copy:
                if j == 'A01test4':
                    continue
                deal = train_val[train_val['bj']==j]
                deal = deal.sample(frac=1)
                if val_count >= round(tot *(val_size+test_size)) :
                    break
                if val_count + deal.shape[0] > round(tot *(val_size+test_size)):
                    if round(tot *(val_size+test_size)) - val_count < 0.5*deal.shape[0] and 'test' in j:
                    # 有数据溢出了，但是不管
                        split_flag = False
                        print(f'类型{index}验证集存在数据泄露，泄露{round(tot *(val_size+test_size)) - val_count}个，泄露种类为{j}')
                    else:
                        continue
                jl_test.append(j)
                jl.remove(j)
                for i in range(deal.shape[0]):
                    if split_flag:
                        # 删除数据，放到dataset/other文件夹中
                        shutil.move('dataset/train/{}/'.format(index)+deal['filename'].values[i],'dataset/other/{}/'.format(index))
                        # 从现有val数据集中随机选取一张图片进行复制
                        val_files = os.listdir('dataset/val/{}/'.format(index))
                        # val_files如果存在含有“copy”的文件，则剔除出val_files列表
                        # 先复制一份val_files，因为在迭代过程中不能修改val_files，否则会出问题
                        val_files_copy = val_files.copy()
                        for i in val_files_copy:
                            if 'copy' in i :
                                origin_file = i[:-8]+'.png'
                                val_files.remove(origin_file) # 删除被复制的原文件
                                if index not in ['08','09']:
                                    val_files.remove(i)
                        if index in ['08','09']:
                            # 进行随机平移和水平翻转
                            transform = transforms.Compose([
                                                            # 随机平移图像,degrees=0表示不旋转,translate=(0.2, 0.2)表示在水平和垂直方向上平移的最大距离
                                                            transforms.RandomAffine(degrees=0,translate=(0.1, 0.1),fill=(255,255,255,255)),
                                                            # 随机水平翻转图像
                                                            transforms.RandomHorizontalFlip(p=0.8),
                                                        ])
                        else:
                            # 进行随机平移
                            transform = transforms.RandomAffine(degrees=0,translate=(0.1, 0.1),fill=(255,255,255,255))
                        val_file = random.choice(val_files)
                        val_new_file = val_file[:-4] + 'copy' + '.png'
                        # 读取图片
                        img = Image.open('dataset/val/{}/'.format(index)+val_file)
                        # 进行随机平移
                        img = transform(img)
                        # 保存图片
                        img.save('dataset/val/{}/'.format(index)+val_new_file)
                        val_new_bj = "_".join(val_new_file.split("_")[:-1])
                        # 在train_val的'filename'列中添加val_new_file
                        train_val = train_val._append({'filename':val_new_file,'label':index,'bj':val_new_bj},ignore_index=True)
                    else:
                        shutil.move('dataset/train/{}/'.format(index)+deal['filename'].values[i],'dataset/val/{}/'.format(index))
                    val_count = len(os.listdir('dataset/val/{}/'.format(index)))
                    if val_count >= round(tot *(val_size+test_size)):
                        jl_test.remove(j)
                        jl.append(j)
                        break
        # 新建字典val_list和train_list
        val_list = {}
        train_list = {}
        for i in jl:
            train_list[i] = train_val[train_val['bj']==i].shape[0]
        for i in jl_test:
            val_list[i] = train_val[train_val['bj']==i].shape[0]
        # 依次输出val_list和train_list
        print("训练集种类有：",train_list)
        print("验证集种类有：",val_list)
        # 划分测试集
        # 以“test”为关键词对jl_test列表进行排序，将含有“test”的元素放在列表的最前
        jl_test = sorted(jl_test,key=lambda x: ('test' in x, train_val[train_val['bj']==x].shape[0]), reverse=True)
        # 从小到大排序
        # jl_test = sorted(jl_test,key=lambda x:train_val[train_val['bj']==x].shape[0],reverse=False)
        test_count = len(os.listdir('dataset/test/{}/'.format(index)))
        if test_count < round(tot * test_size):
            for j in jl_test:
                deal = train_val[train_val['bj']==j]
                if test_count >= round(tot * test_size):
                    break
                deal = deal.sample(frac=1)
                for i in range(deal.shape[0]):
                    shutil.move('dataset/val/{}/'.format(index)+deal['filename'].values[i],'dataset/test/{}/'.format(index))
                    test_count = len(os.listdir('dataset/test/{}/'.format(index)))
                    if test_count >= round(tot * test_size):
                        break

def move_files_to_folder(source_paths, target_path):
    """
    将源路径列表中的所有文件移动到目标路径下的对应文件夹中。

    参数:
    source_paths: 源路径列表，每个源路径下应包含多个文件夹，每个文件夹中包含多个文件。
    target_path: 目标路径，所有文件将被移动到这个路径下的对应文件夹中。
    """
    for source_path in source_paths:
        for folder in os.listdir(source_path):
            # 如果目标路径下的文件夹不存在就创建
            if not os.path.exists(os.path.join(target_path, folder)):
                os.makedirs(os.path.join(target_path, folder))
            for file in os.listdir(os.path.join(source_path, folder)):
                if "copy" in file:
                    os.remove(os.path.join(source_path, folder, file))
                else :
                    # 将文件移动到目标路径下的对应文件夹中
                    shutil.move(os.path.join(source_path, folder, file), os.path.join(target_path, folder, file))


def get_all_files(source_paths):
    """
    获取源路径列表中的所有文件的路径。

    参数:
    source_paths: 源路径列表，每个源路径下应包含多个文件夹，每个文件夹中包含多个文件。
    返回值:
    一个包含所有文件路径的列表。
    """
    all_files = []
    for source_path in source_paths:
        for folder in os.listdir(source_path):
            for file in os.listdir(os.path.join(source_path, folder)):
                file_path = os.path.join(source_path, folder, file)
                all_files.append(file_path)
    return all_files

def processing_data_main(origin_data_path, label, val_size, test_size, mode,restart_flag=True):
    # restart_flag为重启标志，若为True则重新生成数据集，默认为True
    # 将原始数据的csv文件转移到dataset/train文件夹
    new_data_path = 'dataset/train/'
    # 如果train文件夹不存在，则新建文件夹
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)
    # 原始数据所在文件夹,若输入为空，则默认为'origin_dataset_clean/train_merged/'
    if not origin_data_path:
        origin_data_path = 'origin_dataset_clean/train_merged/'
    # 删除dataset文件夹中的'class_count'txt文本，若不存在则建立空文件
    if os.path.exists('dataset/class_count.txt'):
        open('dataset/class_count.txt', 'w').close()
    # 加载设置参数
    opt = processing_parse_opt()
    # 加载输入变量
    if val_size:
        opt.val_size = float(val_size)
    if test_size:
        opt.test_size = float(test_size)
    if mode:
        if mode == '生产参数归一化':
            opt.pattern = 'mode1'
        elif mode == '普通归一化':
            opt.pattern = 'mode2'
        else:
            opt.pattern = 'mode3'

    if restart_flag:
        # 删除dataset文件夹中的train、test、val文件夹
        shutil.rmtree('dataset/train', ignore_errors=True)
        shutil.rmtree('dataset/test', ignore_errors=True)
        shutil.rmtree('dataset/val', ignore_errors=True)
        shutil.rmtree('dataset/other', ignore_errors=True)
        # 生成图片 224*112
        process_data(origin_data_path,new_data_path,opt.image_size1,opt.image_size2,opt.pattern)


    with open(opt.label_path, 'w+', encoding='utf-8') as f:
        # 如果label为空
        if not label:
            label=['工作正常','供液不足','气体影响','气锁','上碰泵','下碰泵','游动阀关闭迟缓','柱塞脱出泵工作筒','游动阀漏','固定阀漏','砂影响+供液不足','惯性影响+工作正常']
        # 逐行写入label
        f.write('\n'.join(label))

    str_len = len(str(len(os.listdir(opt.data_path))))
    for idx, i in enumerate(os.listdir(opt.data_path)):
        os.rename(r'{}/{}'.format(opt.data_path, i), r'{}/{}'.format(opt.data_path, str(idx).zfill(str_len)))

    if not restart_flag:
        # 将文件都移动到train文件夹中
        move_files_to_folder(['dataset/val', 'dataset/test', 'dataset/other'], 'dataset/train')

    # 统计原始数据集中的各个类别的图像数量
    count_images_and_save('dataset/train/', 'dataset/class_count.txt',label)
    # 数据划分
    split_dataset(opt.val_size, opt.test_size, label)
    # 统计train、test、val文件夹中的各个类别的图像数量
    count_images_and_save('dataset/train/', 'dataset/class_count.txt', label)
    count_images_and_save('dataset/val/', 'dataset/class_count.txt', label)
    count_images_and_save('dataset/test/', 'dataset/class_count.txt', label)

    # 返回所有生成图片的路径
    images = get_all_files(['dataset/train', 'dataset/test', 'dataset/val'])

    # 读取并返回class_count.txt的内容
    with open('dataset/class_count.txt', 'r') as f:
        return f.read(), images











if __name__ == '__main__':

    # 删除dataset文件夹中的train、test、val文件夹
    # shutil.rmtree('dataset/train', ignore_errors=True)
    # shutil.rmtree('dataset/test', ignore_errors=True)
    # shutil.rmtree('dataset/val', ignore_errors=True)
    # shutil.rmtree('dataset/other', ignore_errors=True)
    # 删除dataset文件夹中的'class_count'txt文本，若不存在则建立空文件
    if os.path.exists('dataset/class_count.txt'):
        open('dataset/class_count.txt', 'w').close()

    # 原始数据所在文件夹
    origin_data_path = 'origin_dataset_clean/train_merged/'
    # 将原始数据的csv文件转移到dataset/train文件夹
    new_data_path = 'dataset/train/'
    # 如果train文件夹不存在，则新建文件夹
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # 生成的训练集（有标签）图像所在文件夹
    folder_path_new = 'dataset/train/'
    # 加载设置参数
    opt = processing_parse_opt()

    # 生成图片 224*112
    # process_data(origin_data_path,new_data_path,opt.image_size1,opt.image_size2,opt.pattern)


    with open(opt.label_path, 'w+', encoding='utf-8') as f:

        # 逐行写入"工作正常"、"供液不足"、"气体影响"、"气锁"、"上碰泵"、"下碰泵"、"游动阀关闭迟缓"、"柱塞脱出泵工作筒"、"游动阀漏"、"固定阀漏"、"砂影响+供液不足"、"惯性影响+工作正常"
        label=['工作正常','供液不足','气体影响','气锁','上碰泵','下碰泵','游动阀关闭迟缓','柱塞脱出泵工作筒','游动阀漏','固定阀漏','砂影响+供液不足','惯性影响+工作正常']
        f.write('\n'.join(label))

    str_len = len(str(len(os.listdir(opt.data_path))))

    for idx, i in enumerate(os.listdir(opt.data_path)):
        os.rename(r'{}/{}'.format(opt.data_path, i), r'{}/{}'.format(opt.data_path, str(idx).zfill(str_len)))


    train_path = 'dataset/train'
    val_path = 'dataset/val'
    test_path = 'dataset/test'
    other_path = 'dataset/other'
    # 将文件都移动到train文件夹中
    move_files_to_folder([val_path, test_path, other_path], train_path)

    # 统计原始数据集中的各个类别的图像数量
    count_images_and_save('dataset/train/', 'dataset/class_count.txt',label)

    # # 数据扩充，（A01 工作正常 --旋转5~15°--> A12 惯性影响+工作正常 这个没做！）
    # augment_images('dataset/train/')

    # # 统计数据增强后的各个类别的图像数量
    # count_images_and_save('dataset/train', 'dataset/class_count.txt')

    # 数据划分
    split_dataset(opt.val_size, opt.test_size,label)

    # 统计train、test、val文件夹中的各个类别的图像数量
    count_images_and_save('dataset/train/', 'dataset/class_count.txt',label)
    count_images_and_save('dataset/val/', 'dataset/class_count.txt',label)
    count_images_and_save('dataset/test/', 'dataset/class_count.txt',label)
    
    # 划分完毕
    print('程序结束！')


    #     # os.chdir(opt.data_path)

#     # for i in os.listdir(os.getcwd()):
#     #     base_path = os.path.join(os.getcwd(), i)
#     #     base_arr = os.listdir(base_path)
#     #     np.random.shuffle(base_arr)
#     #     # 验证集
#     #     val_path = base_path.replace('train', 'val')
#     #     if not os.path.exists(val_path):
#     #         os.makedirs(val_path)
#     #     val_need_copy = base_arr[int(len(base_arr) * (1 - opt.val_size - opt.test_size)):int(len(base_arr) * (1 - opt.test_size))]
#     #     for j in val_need_copy:
#     #         shutil.move(os.path.join(base_path, j), os.path.join(val_path, j))
#     #     # 测试集
#     #     test_path = base_path.replace('train', 'test')
#     #     if not os.path.exists(test_path):
#     #         os.makedirs(test_path)
#     #     test_need_copy = base_arr[int(len(base_arr) * (1 - opt.test_size)):]
#     #     for j in test_need_copy:
#     #         shutil.move(os.path.join(base_path, j), os.path.join(test_path, j))

#     # # 切换到dataset文件夹
      #   os.chdir(r'../') # 切换到上一级目录
#     # print('数据集划分完毕！')