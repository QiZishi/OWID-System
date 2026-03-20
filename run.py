import gradio as gr
import random
import pandas as pd
import zipfile
import os,re
import shutil
import patoolib
from time import sleep
from processing import processing_data_main
from main import train_main
from metrice import metrice_main
from predict import predict_main
import time
import numpy as np
import tqdm
import json
import pickle
import joblib
# 导入全局变量中断标志，默认为False
from mainweb_globalValue import stop_training_flag
# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import dashscope

dashscope.api_key=os.environ.get("DASHSCOPE_API_KEY", "")


'''''Theme主题设置区'''
# 从本地文件加载主题 使用xiaobaiyuan的主题
my_theme = gr.Theme.load('websystem_data/网页主题/themes_theme_schema@0.0.2.json')
''''界面函数区'''

# 功能介绍函数
def chat(message,history):
    history = history or []
    flag = True
    response_generator = dashscope.Generation.call(
        model='model-name',
        prompt=message,
        stream=True,
        top_p=0.8)
    if message.startswith('介绍示功图生成模块'):
        with open('websystem_data/示功图生成模块介绍.txt', 'r',encoding='utf-8') as f:
            response = f.read()
            yield response
    elif message.startswith('介绍工控系统网络诊断模块'):
        with open('websystem_data/工控系统网络诊断模块介绍.txt', 'r',encoding='utf-8') as f:
            response = f.read()
            yield response
    elif message.startswith('介绍抽油机井工况诊断模块'):
        with open('websystem_data/抽油机井工况诊断模块介绍.txt', 'r',encoding='utf-8') as f:
            response = f.read()
            yield response
    elif message.startswith('介绍抽油机井工况识别模型训练模块'):
        with open('websystem_data/抽油机井工况识别模型训练模块介绍.txt', 'r',encoding='utf-8') as f:
            response = f.read()
            yield response
    elif message.startswith('介绍抽油机井工况识别模型性能评估模块'):
        with open('websystem_data/抽油机井工况识别模型性能评估模块介绍.txt', 'r',encoding='utf-8') as f:
            response = f.read()
            yield response
    elif message.startswith('介绍用户管理模块'):
        response = "用户管理模块提供新用户注册、已有用户信息更新服务（仅限具有管理员身份的账户进行操作）。"
        yield response
    else:
        for resp in response_generator:
            paragraph = resp.output['text']
            yield paragraph

# 示功图生成函数app2_fn
def app2_fn(zip_file, val_size, test_size, mode, regenerate_flag):
    # 将regenerate_flag转换为bool类型
    regenerate_flag = True if regenerate_flag == "是" else False
    if val_size + test_size > 1:
        raise gr.Error("划分比例不正确!请重新设置！")
        # return "划分比例不正确", []
    # 如果上传的zip_file文件不为.zip、.tar、.rar、.7z格式之一，则报错
    if zip_file is not None and not (zip_file.name.endswith('.zip') or zip_file.name.endswith('.tar') or zip_file.name.endswith('.rar') or zip_file.name.endswith('.7z')):
        raise gr.Error("上传的文件格式不正确，请重新上传！")
        # return "上传的文件格式不正确，请重新上传！", []
    # 如果压缩文件不为空
    if zip_file is not None:
        # 提取压缩文件包名字
        zip_file_name = zip_file.name
        base_name = os.path.basename(zip_file_name)  # 获取文件名，例如 "myfile.zip"
        file_name_zip = os.path.splitext(base_name)[0]  # 去掉文件扩展名，例如 "myfile"
        # 解压压缩文件，支持中文文件名,支持zip、rar、7z格式
        origin_data_path = "websystem_data/user_dataset/"
        if not os.path.exists(origin_data_path):
            os.makedirs(origin_data_path)
        patoolib.extract_archive(zip_file.name, outdir=origin_data_path)
        # 原始数据所在文件夹
        origin_data_path = 'websystem_data/user_dataset/' + file_name_zip + '/'
    else:
        # 原始数据所在文件夹,若输入为空，则默认为'origin_dataset_clean/train_merged/'
        origin_data_path = 'origin_dataset_clean/train_merged/'

    # 读取data_folder中的第一级文件夹作为label
    label = [f for f in os.listdir(origin_data_path) if os.path.isdir(os.path.join(origin_data_path, f))]
    # 如果label中含有英文字符
    if not any([contains_chinese(l) for l in label]):
        label=['工作正常','供液不足','气体影响','气锁','上碰泵','下碰泵','游动阀关闭迟缓','柱塞脱出泵工作筒','游动阀漏','固定阀漏','砂影响+供液不足','惯性影响+工作正常']

    output_text,images = processing_data_main(origin_data_path, label,val_size, test_size, mode, regenerate_flag)
    # 清除原始数据
    if zip_file is not None:
        shutil.rmtree("websystem_data/user_dataset")
    # 清除dataset/other文件夹
    shutil.rmtree('dataset/other')
    # 将dataset文件夹压缩成一个压缩包
    shutil.make_archive('dataset', 'zip', 'dataset')
    # 将生成的压缩包移动到指定文件夹
    shutil.move('dataset.zip', 'websystem_data/dataset.zip')
    # 返回生成的压缩包路径
    output_zip = 'websystem_data/dataset.zip'
    return  output_zip,output_text, images

# 模型训练函数app3_fn
def app3_fn(model_choice, use_pretrained, batch_size, epochs, use_mixed_precision, metrics, early_stopping, optimizer, loss,lr_schedule, learning_rate, lr_warmup, warmup_epochs):
    # 运行函数时，默认将中断训练标志设置为False
    global stop_training_flag
    stop_training_flag = False
    # 将use_pretrained转换为bool类型
    if use_pretrained == "是":
        use_pretrained = True
    else:
        use_pretrained = False
    # 将use_mixed_precision转换为bool类型
    if use_mixed_precision == "是":
        use_mixed_precision = True
    else:
        use_mixed_precision = False
    # 将lr_warmup转换为bool类型
    if lr_warmup == "是":
        lr_warmup = True
    else:
        lr_warmup = False
    # 将metrics转换为对应的字符串
    if metrics == "损失":
        metrics = "loss"
    elif metrics == "准确率":
        metrics = "acc"
    else:
        metrics = "mean_acc"
    # 开始时间
    starttime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # 设置save_path
    save_path = "websystem_data/trained_model/"+model_choice+'-'+starttime
    # 调动train_main这个生成器
    train_main_gen = train_main(str(model_choice), use_pretrained, int(batch_size), int(epochs),
                                save_path,loss,optimizer,lr_schedule,float(learning_rate),
                                use_mixed_precision, lr_warmup, float(warmup_epochs),
                                metrics, int(early_stopping))

    for df in train_main_gen:
        # 训练记录文字
        # 打开训练过程记录文件
        with open(os.path.join(save_path, '训练过程记录.txt'), 'r') as f:
            output_text = f.read()
        # 分解训练记录数据
        # 从df中分出epoch和lr
        df_lr = df[['epoch', 'lr']]
        # 从df中分出epoch和train_acc和test_acc
        df_acc = pd.DataFrame({'epoch': df['epoch'], 'acc': df['train_acc'], '类别': ['训练集准确率']*df.shape[0]})
        df_acc = df_acc._append(pd.DataFrame({'epoch': df['epoch'], 'acc': df['test_acc'], '类别': ['测试集准确率']*df.shape[0]}))
        # 从df中分出epoch和train_mean_acc和test_mean_acc
        df_mean_acc = pd.DataFrame({'epoch': df['epoch'], 'mean_acc': df['train_mean_acc'], '类别': ['训练集平均准确率']*df.shape[0]})
        df_mean_acc = df_mean_acc._append(pd.DataFrame({'epoch': df['epoch'], 'mean_acc': df['test_mean_acc'], '类别': ['测试集平均准确率']*df.shape[0]}))
        # 从df中分出epoch和train_loss和test_loss，注意不要epoch为0的数据
        df = df.iloc[1:]
        df_loss = pd.DataFrame({'epoch': df['epoch'], 'loss': df['train_loss'], '类别': ['训练集损失']*df.shape[0]})
        df_loss = df_loss._append(pd.DataFrame({'epoch': df['epoch'], 'loss': df['test_loss'], '类别': ['测试集损失']*df.shape[0]}))
        # 如果中断训练标志为True，则中断训练
        if stop_training_flag:
            # 删除save_path文件夹
            shutil.rmtree(save_path)
            # 弹出模型中止训练的警告
            gr.Warning('模型训练中止！')
            yield output_text+'\n模型训练中止!\n',df_lr,df_acc,df_mean_acc,df_loss
            break
        # 返回训练日志和训练图表
        yield  output_text,df_lr,df_acc,df_mean_acc,df_loss

def stop_training():
    # 设置中断训练标志为True，使训练中断
    global stop_training_flag
    stop_training_flag = True


# 更新模型选择下拉框（用于模型验证和模型测试）
def save_path_model():
    model_list = [os.path.basename(entry.path) for entry in os.scandir('websystem_data/best_model') if entry.is_dir()]
    model_list.extend([os.path.basename(entry.path) for entry in os.scandir('websystem_data/trained_model') if entry.is_dir()])
    # 返回新下拉框
    return model_list

# 模型验证函数app4_fn
def app4_fn(save_path_in,task_in,batch_size_in,tsne_in,half_in):

    # 将tsne_in转换为bool类型
    if tsne_in == "是":
        tsne_in = True
    else:
        tsne_in = False
    # 将half_in转换为bool类型
    if half_in == "是":
        half_in = True
    else:
        half_in = False
    # 将task_in转换为对应的字符串
    if task_in == "测试集":
        task_in = "tset"
    elif task_in == "验证集":
        task_in = "val"
    else:
        task_in = "train"
    # 如果save_path_in中含字符'-'
    if '-' in save_path_in:
        # 用户训练模型
        save_path_in = "websystem_data/trained_model/"+save_path_in
    else:
        # 最优模型
        save_path_in = "websystem_data/best_model/"+save_path_in
    output_save_path = metrice_main(save_path_in,task_in,batch_size_in,tsne_in,half_in)
    # 读取output_save_path中的overall_result.csv和perclass_result.csv，以gbk格式读取，分别转换为两个df
    overall_result = pd.read_csv(output_save_path+'/overall_result.csv',encoding='gbk')
    perclass_result = pd.read_csv(output_save_path+'/perclass_result.csv',encoding='gbk')
    # 读取output_save_path中的Multiclass_PR_Curve.png、Multiclass_ROC_Curve_ALL.png、confusion_matrix.png将它们的路径存放到一个名为output_images的列表中
    output_images = [(output_save_path+'/confusion_matrix.png','混淆矩阵'),(output_save_path+'/Multiclass_ROC_Curve_ALL.png','ROC曲线'),(output_save_path+'/Multiclass_PR_Curve.png','PR曲线')]
    if tsne_in:
        # 读取output_save_path中的tsne.png将它的路径存放到output_images的列表中
        output_images.append((output_save_path+'/tsne.png','TSNE可视化'))
    # 返回overall_result、perclass_result、output_images
    return overall_result,perclass_result,output_images


# 模型测试页面部分控件显示/隐藏函数
def file_in_change(file_in):
    if file_in == "单张图片":
        return gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)

    else:
        return  gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),gr.update(visible=True)

# 模型测试函数app_fn
def app_fn(source_in_pic,source_in,file_in, save_path_in, cam_visual_in, cam_type_in, half_in):
    # 将file_in转换为bool类型
    if file_in == "单张图片":
        file_in = True
    else:
        file_in = False
            # 将cam_visual_in转换为bool类型
    if cam_visual_in == "是":
        cam_visual_in = True
    else:
        cam_visual_in = False
    # 将half_in转换为bool类型
    if half_in == "是":
        half_in = True
    else:
        half_in = False
    # 如果save_path_in中含字符'-'
    if '-' in save_path_in:
        # 用户训练模型
        save_path_in = "websystem_data/trained_model/"+save_path_in
    else:
        # 最优模型
        save_path_in = "websystem_data/best_model/"+save_path_in
    # 如果上传的是图片（上传图片用image控件，type为pil）
    if file_in:
        if source_in_pic:
            # 将图片保存到指定文件夹
            source_in = source_in_pic
        else:
            raise gr.Error("未检测到图片上传！请上传图片后再点击运行！")
    else:
        # 如果上传的source_in文件不为.zip、.tar、.rar、.7z格式之一，则报错
        if source_in is not None and not (source_in.name.endswith('.zip') or source_in.name.endswith('.tar') or source_in.name.endswith('.rar') or source_in.name.endswith('.7z')):
            raise gr.Error("上传的文件格式不正确，请重新上传！")
        # 如果压缩文件不为空
        if source_in is not None:
            # 提取压缩文件包名字
            zip_file_name = source_in.name
            base_name = os.path.basename(zip_file_name)  # 获取文件名，例如 "myfile.zip"
            file_name_zip = os.path.splitext(base_name)[0]  # 去掉文件扩展名，例如 "myfile"
            # 解压压缩文件，支持中文文件名,支持zip、rar、7z格式
            origin_data_path = "websystem_data/user_pred_dataset/"
            if not os.path.exists(origin_data_path):
                os.makedirs(origin_data_path)
            patoolib.extract_archive(source_in.name, outdir=origin_data_path)
            # 原始测试数据所在文件夹
            source_in = 'websystem_data/user_pred_dataset/' + file_name_zip + '/'
        else:
            # 原始测试数据所在文件夹,若输入为空，则默认为'websystem_data/predict_dataset/'
            source_in = 'websystem_data/predict_dataset/'

    if file_in:
        # 对单张图像进行预测
        output_save_path,labels,pred_result,pred = predict_main(source_in, save_path_in, cam_visual_in, cam_type_in, half_in)
        # 输出结果
        result_dict ={labels[i]: float(pred_result[i]) for i in range(len(labels))}
        # 读取'websystem_data/故障分析.xlsx'文件作为output_df
        output_df = pd.read_excel('websystem_data/故障分析.xlsx')
        # 根据labels[pred]找到对应故障分析
        output_text = output_df[output_df['故障类型'] == labels[pred]]['故障分析'].values[0]
        return result_dict,output_text,None,None,None
    else:
        # 对目录中的每一张图像进行预测
        output_save_path= predict_main(source_in, save_path_in, cam_visual_in, cam_type_in, half_in)
        # 读取output_save_path中的overall_result.csv和perclass_result.csv，以gbk格式读取，分别转换为两个df
        output_result = pd.read_excel(output_save_path+'/抽油机井工况诊断结果.xlsx')
        # 读取output_save_path中的所有图片，将它们的路径和文件名字放在一个元组里，然后组成一个列表
        output_images = [(os.path.join(output_save_path, f), f) for f in os.listdir(output_save_path) if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']]
        # 清除原始数据
        if os.path.exists("websystem_daa/user_pred_dataset"):
            # 清除原始测试数据
            shutil.rmtree("websystem_data/user_pred_dataset")
        # 将output_save_path文件夹压缩成一个压缩包
        shutil.make_archive('抽油机井工况诊断结果', 'zip', output_save_path)
        # 将生成的压缩包移动到指定文件夹
        shutil.move('抽油机井工况诊断结果.zip', 'websystem_data/抽油机井工况诊断结果.zip')
        # 返回生成的压缩包路径
        output_zip = 'websystem_data/抽油机井工况诊断结果.zip'
        # 返回overall_result、perclass_result、output_images
        return None,None,output_zip,output_result,output_images


# 检测中文字符
def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))

# 用户管理
def register(login_account, login_password, register_account, register_password, register_role):
    # 读取xlsx文件
    df = pd.read_excel('websystem_data/users.xlsx')
    # 检查输入的账号和密码是否在xlsx文件中，且对应身份为administer
    if ((df['用户名'] == login_account) & (df['密码'] == login_password) & (df['身份'] == '管理员')).any():
        # 检查是否输入了注册账号和注册密码
        if not register_account or not register_password:
            # 返回管理员身份验证成功和xlsx文件内容
            return login_account+'管理员身份验证成功,欢迎您！', df
        # 如果注册账号或密码中含有不是大小写字母和数字的字符，则报错
        if not register_account.isalnum() or contains_chinese(register_account) or not register_password.isalnum() or contains_chinese(register_password):
            gr.Warning('错误：账号或密码中含有非法字符！')
            return '错误：账号或密码中含有非法字符！', pd.DataFrame()
        # 如果注册账号已经存在且注册身份和注册密码不为空，则对df中对应行的数据进行更新
        if (df['用户名'] == register_account).any() and register_role:
            df.loc[df['用户名'] == register_account, '密码'] = register_password
            df.loc[df['用户名'] == register_account, '身份'] = register_role
            df.to_excel('websystem_data/users.xlsx', index=False)
            # 返回更新成功的消息和xlsx文件的内容
            return '用户'+ register_account +'信息更新成功！身份已更新为：'+register_role+"。", df
        # 将输入的注册账号和密码以及下拉框选择的身份写入xlsx文件
        new_user = pd.DataFrame([[register_account, register_password, register_role]], columns=df.columns)
        df = df._append(new_user, ignore_index=True)
        df.to_excel('websystem_data/users.xlsx', index=False)
        # 返回注册成功的消息和xlsx文件的内容
        return '新用户'+ register_account +'注册成功！', df
    else:
        # 返回错误警告
        gr.Warning('错误：账号或密码错误，或者你没有管理员权限！')
        return '错误：账号或密码错误，或者你没有管理员权限！', pd.DataFrame()



# 功能介绍随机开场白函数
def chat_value():
    # 随机开场白
    chat_value_list = [
                    '你好，我是你的智能助手🤖。试试对我说:“介绍用户管理模块。”',
                    '你好，我是你的智能助手🤗。试试对我说:“介绍抽油机井工况诊断模块。”',
                    '你好，我是你的智能助手😁。试试对我说:“介绍示功图生成模块。”',
                    '你好，我是你的智能助手😉。试试对我说:“介绍抽油机井工况识别模型训练模块。”',
                    '你好，我是你的智能助手🥰。试试对我说:“介绍抽油机井工况识别模型性能评估模块。”',
                    '你好，我是你的智能助手🤩。试试对我说:“介绍工控系统网络诊断模块。”'
                    ]
    return random.choice(chat_value_list)

# 工控系统网络流量检测函数app_fn_ml
def app_fn_ml(file,model):
    # 如果file为空
    if file is None:
        # 使用默认数据集
        file = 'websystem_data/ml_test/10_gas_final.csv'
    # 判断file是xlsx文件还是csv文件
    if file.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.endswith('.csv'):
        df = pd.read_csv(file)
    # 从websystem_data\ml_test\selected_features.json中加载特征选择结果
    with open('websystem_data/ml_test/selected_features.json', 'r') as f:
        selected_features = json.load(f)
    # 从websystem_data\ml_test\scaler.pkl中加载scaler
    with open('websystem_data/ml_test/stdScale.pkl', 'rb') as f:
        scaler = pickle.load(f)
    # 将df的特征列和标签列分开
    X = df[selected_features]
    y = df['label']
    # 对X进行归一化
    X = scaler.transform(X)
    # 根据model选择模型
    with open('websystem_data/ml_test/run_ml/'+model+'/best.pkl', 'rb') as f:
        model = joblib.load(f)
    # 使用模型进行预测
    y_pred = model.predict(X)
    # 将y_pred的值转换为字符串
    y_pred = [str(i) for i in y_pred]
    # 创建一个映射
    label_mapping = {
        '0': '正常样本',
        '1': '普通恶意响应注入攻击',
        '2': '复杂恶意响应注入攻击',
        '3': '恶意状态命令注入攻击',
        '4': '恶意参数命令注入攻击',
        '5': '恶意功能命令注入攻击',
        '6': '拒绝服务攻击',
        '7': '侦察攻击'
    }
    # 使用映射替换y_pred的值
    y_pred = np.array([label_mapping[str(label)] for label in y_pred])
    # 将预测结果转换为DataFrame，第一列列名为序号，是y_pred的索引，第二列列名为预测结果，是y_pred的值
    y_pred_df = pd.DataFrame(y_pred, columns=['诊断结果'])
    y_pred_df.index.name = '样本序号'
    y_pred_df.reset_index(inplace=True)
    # 将y_pred_df的数据保存到一个名为predict_result.xlsx的文件中
    y_pred_df.to_excel('websystem_data/ml_test/诊断结果.xlsx', index=False)
    # 返回预测结果的路径
    output = 'websystem_data/ml_test/诊断结果.xlsx'
    return y_pred_df,output











'''''*********************************************************************************'''

'''界面设置区'''


# 功能介绍
def app():
    with gr.Blocks() as app:
        gr.Markdown("### 🤖智能助手\n-网页介绍：搭载大模型的智能助手🤖竭诚为您介绍本系统各模块功能及相关油气知识！")

        # 聊天机器人组件：初始对话中放一条提示
        chatbot = gr.Chatbot(value=[[None, chat_value()]], label="🤖 智能助手")

        # 输入区域：一个文本框和一个“发送”按钮
        with gr.Row():
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题...")
            send_btn = gr.Button("发送")

        # 定义处理函数，将新消息追加到 history
        def new_chat(user_message, chat_history):
            # chat_history 是当前的 (用户, AI) 对话列表
            updated_history = chat(user_message, chat_history)
            # 确保返回的对话历史记录是一个包含长度为2的列表的列表
            updated_history = [[msg[0], msg[1]] for msg in updated_history]
            # 返回给前端：清空输入框("") 和 更新后的对话
            return user_message, updated_history

        # 事件绑定：回车 和 点击按钮 都执行 new_chat
        user_input.submit(new_chat, [user_input, chatbot], [user_input, chatbot])
        send_btn.click(new_chat, [user_input, chatbot], [user_input, chatbot])
# 工控系统网络流量测试
def app_ml():
    with gr.Blocks() as app:
        # 默认不展开介绍
        with gr.Accordion("工控系统网络诊断模块(展开了解详细介绍)",open=False):
            gr.Markdown('''
    ##### -功能介绍
    工控系统网络诊断模块提供工控系统网络诊断服务，
    用户可以上传自己的工控系统网络流量数据集，或使用系统默认工控系统网络流量数据集，系统会自动进行网络诊断。
    本系统强烈建议用户先进行工控系统网络诊断，在确保控制系统未受到网络攻击的情况下再进行抽油机井工况诊断，以确保抽油机井工况数据未受到网络攻击污染，确保工况数据的真实可靠性！

    ##### 诊断数据集获取
    1.用户可以上传自己的工控系统网络流量数据集，或使用系统默认工控系统网络流量数据集，数据集文件格式要求为.csv或.xlsx文件。
    ##### 模型选择
    1.用户可使用系统已经训练完毕的机器学习模型进行工控系统网络安全诊断。

    2.模型选择完毕后，点击运行按钮，系统进入工控系统网络诊断状态。

    ##### 诊断结果展示

    1.系统会返回诊断结果统计表并提供文件下载链接。

    ''')
        with gr.Row():
            # 输入
            with gr.Column():
                # 上传文件
                source_path_in = gr.File(label="上传网络流量数据（默认使用系统自带数据集，文件格式为.csv/.xlsx）", type='filepath', interactive=True)
                # 选择模型
                model = gr.Dropdown(label="模型选择", choices=['GA_LightGBM' ,'LightGBM','AdaBoost', 'RandomForest',  'MLP',
                                                            'SVM'], value='GA_LightGBM')
            # 输出
            with gr.Column():
                # 上传图片
                with gr.Accordion("诊断结果"):
                    # 输出结果
                    output_result = gr.Dataframe()
                    # 提供下载
                    output_zip = gr.File(label="下载诊断结果统计表",scale = 0.2)


        with gr.Row():
            gr.Button("▶️运行",variant='primary').click(app_fn_ml, inputs=[source_path_in,model],outputs=[output_result,output_zip])
    return app



# 模型测试
def app1():
    with gr.Blocks() as app:
    # 默认不展开介绍
        with gr.Accordion("抽油机井工况诊断模块(展开了解详细介绍)",open=False):
            gr.Markdown('''
##### -功能介绍
抽油机井工况诊断模块提供抽油机井工况诊断服务，
用户可以上传自己的单张示功图图片（或示例图片中的图片）或上传图片压缩包（若不上传图片压缩包，默认使用系统自带测试数据）对系统已经训练完毕的模型
（或在工况识别模型训练模块自行训练完毕的模型）进行工况诊断，系统会自动进行工况诊断。
如果用户上传的是单张图片，则系统返回工况诊断类型、故障原因及解决方案。如果用户上传的是图片压缩包，则系统返回工况诊断结果压缩包（含工况诊断结果统计表和工况诊断结果图）、工况诊断结果统计表和工况诊断结果图。
##### 诊断数据集获取
1.用户可通过点击“上传文件类型”单选框控件进行上传文件类型选择，默认为单张图片；

2.用户选择上传文件类型为单张图片时，用户可自行上传示功图图片（要求为224*224大小的RGB图片）,同时用户也可以点击示例图片中的图片（共有12张，每页6张，对应系统训练数据集的十二种工况）进行工况诊断；

3.用户选择上传文件类型为图片压缩包时，用户可自行上传示功图图片压缩包，压缩包内图片要求为224*224大小的RGB图片，压缩包内图片数量不限，所有图片放在同一个文件夹里，同时用户也可以选择不上传压缩包，系统会自动选择默认诊断数据集压缩包进行工况诊断。
##### 模型选择及参数
1.用户可使用系统已经训练完毕的模型进行工况诊断，也可以使用自己在工况识别模型训练模块训练到的模型进行工况诊断，在模型选择下拉框中，模型名字+具体日期的选项为用户自行训练的模型，没有日期后缀的选项为系统原生模型；

2.在性能评估参数设置方面，系统默认不使用F16半精度推理，默认不使用热力图可视化，当使用热力图可视化的时候，用户可选择热力图算法。

3.上述参数设置完毕后，点击运行按钮，系统进入工况诊断状态。

##### 诊断结果展示

1.在上传单张照片模式下，系统会返回工况诊断类型、故障原因及解决方案；

2.在上传图片压缩包模式下，系统会返回工况诊断结果压缩包（含工况诊断结果统计表和工况诊断结果图）、工况诊断结果统计表和工况诊断结果图，用户可以点击下载按钮下载压缩包，也可以点击图片查看大图并下载原图。
''')
        with gr.Row():
            # 输入
            with gr.Column():
                # 上传图片
                source_path_in_pic = gr.Image(label="上传图片", type='filepath', interactive=True)
                # 上传压缩包
                source_path_in = gr.File(label="上传图片压缩包（默认使用系统自带诊断数据集）", type='filepath', interactive=True,visible=False)
                # 选择上传文件类型
                file_in = gr.Radio(label="上传文件类型（默认上传单张图片）", choices=["单张图片", "图片压缩包（支持.zip、.tar、.rar、.7z格式）"], value="单张图片")
                with gr.Accordion("工况诊断参数设置"):
                    with gr.Row():
                        save_path_in = gr.Dropdown(label="模型选择", choices=save_path_model(), value='repconvnest')
                        half_in = gr.Radio(label="是否使用F16半精度推理（默认不使用）", choices=["是", "否"], value="否")
                    with gr.Row():
                        cam_visual_in= gr.Radio(label="是否使用热力图可视化（默认不使用）", choices=["是", "否"], value="否")
                        cam_type_in = gr.Dropdown(label="热力图类型（默认使用GradCAMPlusPlus）", choices=['GradCAM', 'HiResCAM', 'ScoreCAM', 'GradCAMPlusPlus', 'AblationCAM', 'XGradCAM', 'EigenCAM', 'FullGrad'], value="GradCAMPlusPlus")
                # gr.Markdown("图片示例")
                gr.Examples(
            examples = [os.path.join(os.path.dirname(__file__), "websystem_data", "examples", f) for f in os.listdir(os.path.join(os.path.dirname(__file__), "websystem_data", "examples"))],
            inputs = source_path_in_pic,
            examples_per_page = 6,
            label="示例图片",)
            # 输出
            with gr.Column():
                # 上传图片
                with gr.Accordion("工况诊断结果"):
                    # output_save_path = gr.Image(label="工况诊断结果", type='filepath',height=400,scale=0.5, interactive=False)
                    result_dict = gr.Label(label="工况诊断结果",num_top_classes=5)
                    output_text = gr.Textbox(label="故障原因及解决方案", show_copy_button=True,lines=3,max_lines=5)
                    # 上传压缩包
                    output_zip = gr.File(label="下载工况诊断结果压缩包（包含工况诊断结果图和诊断结果统计表）",scale = 0.2,visible=False)
                    output_result = gr.Dataframe(label='-工况诊断结果统计-',visible=False)
                    output_images = gr.Gallery(label="工况诊断结果可视化", show_label=True,columns=[2], rows=[1], object_fit="contain", height=400,visible=False)

        with gr.Row():
            gr.Button("▶️运行",variant='primary').click(app_fn, inputs=[source_path_in_pic,source_path_in,file_in, save_path_in, cam_visual_in, cam_type_in, half_in],
                                    outputs=[result_dict,output_text,output_zip,output_result,output_images])
        file_in.change(fn=file_in_change,inputs=[file_in],outputs=[source_path_in_pic,result_dict,output_text,
                                                                source_path_in,output_zip,output_result,output_images])
    return app

# 示功图生成
def app2():
    with gr.Blocks() as app:
        # 默认不展开介绍
        with gr.Accordion("示功图生成模块(展开了解详细介绍)",open=False):
            gr.Markdown('''
    ##### 功能介绍
    示功图生成模块是一个用于生成示功图数据集的工具，用户可以上传自己的示功图数据集，也可以使用系统自带的示功图数据集。系统会自动对数据集进行预处理，划分训练集、验证集和测试集，并生成示功图。用户可以下载生成的示功图数据集压缩包，里面包含划分好的训练集、验证集、测试集、标签文件和样本数量统计。
    ##### 数据集上传要求
    1.数据集文件应该为csv文件，且包含以下两列数据：位移和载荷，列名必须为位移(m)和载荷(kn)；

    2.数据集csv文件命名应该为类似A01_20_120.csv的形式，A01为类别名，20表示该文件里有20个样本，120为单个样本采样长度；

    3.数据集文件夹格式应符合如下标准：

        数据集/类别1/*.csv

        数据集/类别2/*.csv

        数据集/类别3/*.csv
        ……；
    4.数据集文件夹需要以压缩包形式上传，本系统支持.zip、.tar、.rar、.7z格式的压缩包；

    5.未上传数据集压缩包，则默认使用系统自带数据集。

    ##### 数据预处理及数据划分

    1.系统自动对数据样本进行异常值处理过滤和删除重复值，默认使用生产参数归一化对数据进行预处理，生成的图片大小为224*224；

    2.数据集划分比例默认为训练集：验证集：测试集=8:2:0，当检测到验证集比例与测试集比例之和大于1系统会报警并中断后续操作直至正确设置划分比例;

    3.系统默认每次运行都会重新生成示功图，如果选择不重新生成示功图，那么系统只会对数据集重新进行划分。

    ##### 生成结果预览及下载
    1.用户可以下载生成的示功图数据集压缩包（内含划分好的训练集、验证集、测试集、标签文件和样本数量统计）；

    2.系统自动展示样本数量统计结果；

    3.系统会自动展示生成的示功图图片，用户可以点击图片查看大图并下载原图。


    ''')
        with gr.Row():
            with gr.Column():
                source_in = gr.File(label="上传原始数据压缩包（支持.zip、.tar、.rar、.7z格式，默认使用系统自带数据集）", type='filepath', interactive=True)
                with gr.Accordion("参数设置"):
                    with gr.Row():
                            val_ratio = gr.Slider(minimum=0, maximum=1, step=0.1, label="验证集划分比例（默认为0.2）", value=0.2)
                            test_ratio = gr.Slider(minimum=0, maximum=1, step=0.1, label="测试集划分比例（默认为0.0）", value=0.0)
                    with gr.Row():
                            img_mode = gr.Radio(choices=["生产参数归一化", "普通归一化", "不归一化"], label="图像处理模式（默认生产参数归一化）", value="生产参数归一化")
                            # 输出的不是bool，要转换
                            regenerate = gr.Radio(label="是否重新生成示功图（默认重新生成）", choices=["是", "否"], value="是")
            with gr.Column():
                        output_zip = gr.File(label="下载生成的数据集压缩包（包含训练集、验证集、测试集、标签和样本数量统计）",scale = 0.2)
                        with gr.Accordion("结果查看"):
                            output_text = gr.Textbox(label="数据集数量统计", lines=3,max_lines=5,show_copy_button=True)
                            output_images = gr.Gallery(label="生成示功图预览", show_label=True,rows = 1,height=200)


        gr.Button("▶️运行",variant='primary').click(app2_fn, inputs=[source_in, val_ratio, test_ratio, img_mode, regenerate], outputs=[output_zip, output_text,output_images])
    return app

def app3():
    with gr.Blocks() as app:
        # 默认不展开介绍
        with gr.Accordion("抽油机井工况识别模型训练模块(展开了解详细介绍)", open=False):
            gr.Markdown('''
    ##### -功能介绍
    抽油机井工况识别模型训练模块提供端到端的模型训练服务，用户可以使用在示功图生成模块中生成的数据集进行模型训练，也可使用系统默认数据集进行训练（用户自定义数据集需要在示功图生成模块中生成以后才可以用于模型训练，否则默认系统自带数据集），选择模型和相关超参数，
                        系统会自动训练模型并展示训练结果（训练过程日志、学习率曲线图、准确率曲线图、平均准确率曲线图和损失曲线图）。

    ##### 模型选择及参数
    1.本系统提供多种先进的深度学习模型供用户选择，默认使用本系统最强大的RepConvNeSt模型，该模型可加载自监督预训练权重，其他模型可加载ImageNet预训练权重（默认不使用）；

    2.在基础训练参数设置方面，系统允许用户设置批次大小（batch_size）、训练轮数（epoch），默认使用混合精度训练以加快训练速度，默认使用平均准确率作为选择最优模型的参考指标，此外，本系统将早停技术引入到模型训练中，当测试损失在经过早停轮数后仍未出现更低值，则系统结束训练；

    3.在优化器参数设置方面，系统允许用户设置优化器、损失函数、学习率调整策略、学习率、学习率预热和预热比例，预热是在模型以初始学习率开始训练前，先从一个较小学习率开始训练（本系统设置为1e-6），经过预热轮数后逐步上升到初始学习率，从而避免模型陷入局部最优和防止梯度爆炸或消失；

    4.在模型选择及相关参数设置完毕后，点击运行按钮，模型开始训练；

    5.在模型训练过程中，点击中止按钮，则模型训练会被中止。


    ##### 训练结果展示

    1.系统在训练过程中会自动展示训练过程日志，用户可以查看模型选择及相关参数设置，同时可以查看模型每一轮训练的结果和最后的训练用时及模型内存大小；

    2.系统在训练过程中会自动展示学习率曲线图、准确率曲线图、平均准确率曲线图和损失曲线图，图片可供下载，同时系统支持对图片进行放大缩小动态查看。
    ''')

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="模型选择",
                        choices=[
                            'repconvnest','convnextv2_atto','shufflenet_v2_x1_0',
                            'resnest50','vgg16','RepVGG-A0',
                            'efficientnet_b0','efficientnet_v2_s',
                            'mobilenetv2','mobilenetv3_large',
                            'ghostnetv2','repghostnet_1_0x',
                            'fasternet_t0','mobileone_s0',
                            'LA_cnn','fdt_capsnet'
                        ],
                        value='repconvnest'
                    )
                    use_pretrained = gr.Radio(
                        label="是否使用预训练权重（默认不使用）", 
                        choices=["是", "否"], 
                        value="否"
                    )

                with gr.Accordion("基础训练参数设置"):
                    with gr.Row():
                        batch_size = gr.Number(
                            label="批次大小（默认为128）",
                            minimum=1, 
                            value=128
                        )
                        epochs = gr.Number(
                            label="训练轮数（默认为100）",
                            minimum=1, 
                            value=100
                        )
                        use_mixed_precision = gr.Radio(
                            label="是否使用混合精度训练（默认使用）", 
                            choices=["是", "否"], 
                            value="是"
                        )
                    with gr.Row():
                        metrics = gr.Dropdown(
                            label="评价指标选择（默认为平均准确率）", 
                            choices=["损失", "准确率", "平均准确率"], 
                            value="平均准确率"
                        )
                        early_stopping = gr.Number(
                            label="早停轮数（默认为20，以损失为标准）", 
                            minimum=0, 
                            value=20
                        )

                with gr.Accordion("优化器参数设置"):
                    with gr.Row():
                        optimizer = gr.Dropdown(
                            label="优化器（默认为AdamW）", 
                            choices=["SGD", "AdamW", "RMSProp"], 
                            value="AdamW"
                        )
                        loss = gr.Dropdown(
                            label="损失函数（默认为FocalMarginLoss）", 
                            choices=[
                                'FocalMarginLoss', 'CrossEntropyLoss','FocalLoss',
                                'PolyLoss', 'PSoftmaxLoss','CapsuleLoss'
                            ], 
                            value="FocalMarginLoss"
                        )
                    with gr.Row():
                        lr_schedule = gr.Dropdown(
                            label="学习率调整策略（默认为ExponentialLR）", 
                            choices=['CosineAnnealingLR', 'ExponentialLR', 'StepLR'], 
                            value="ExponentialLR"
                        )
                        learning_rate = gr.Number(
                            label="学习率（默认为0.0001）", 
                            value=0.0001, 
                            step=1e-12,
                            minimum=1e-12,
                            maximum=1
                        )
                    with gr.Row():
                        lr_warmup = gr.Radio(
                            label="是否开启学习率预热（默认开启）", 
                            choices=["是", "否"], 
                            value="是"
                        )
                        warmup_epochs = gr.Slider(
                            label="预热比例（默认0.05）",
                            minimum=0,  # Slider 使用 minimum/maximum
                            maximum=0.2,
                            step=0.01,
                            value=0.05,
                            visible=True
                        )

            # ================== 右侧：输出与图表 ==================
            with gr.Column():
                output_text = gr.Textbox(
                    label="训练日志",
                    lines=5,
                    max_lines=6,
                    show_copy_button=True
                )
                with gr.Accordion("训练图表"):
                    with gr.Row():
                        output_plot_lr = gr.ScatterPlot(
                            label="学习率曲线",
                            x="epoch",
                            y="lr",
                            color="类别",
                            connect_points=True,
                            tooltip=["epoch", "lr", "类别"],
                            title="学习率",
                        )
                        output_plot_acc = gr.ScatterPlot(
                            label="准确率曲线",
                            x="epoch",
                            y="acc",
                            color="类别",
                            connect_points=True,
                            tooltip=["epoch", "acc", "类别"],
                            y_range=(0, 1.05),
                            title="准确率曲线",
                        )
                    with gr.Row():
                        output_plot_meanacc = gr.ScatterPlot(
                            label="平均准确率曲线",
                            x="epoch",
                            y="mean_acc",
                            color="类别",
                            connect_points=True,
                            tooltip=["epoch", "mean_acc", "类别"],
                            y_range=(0, 1.05),
                            title="平均准确率",
                        )
                        output_plot_loss = gr.ScatterPlot(
                            label="损失曲线",
                            x="epoch",
                            y="loss",
                            color="类别",
                            connect_points=True,
                            tooltip=["epoch", "loss", "类别"],
                            title="损失",
                        )

        # =============== 按钮区 ===============
        with gr.Row():
            # 开始训练
            gr.Button("▶️开始训练", variant='primary').click(
                fn=app3_fn, 
                inputs=[
                    model_dropdown, use_pretrained, batch_size, epochs, 
                    use_mixed_precision, metrics, early_stopping, optimizer,
                    loss, lr_schedule, learning_rate, lr_warmup, warmup_epochs
                ],
                outputs=[
                    output_text, output_plot_lr, output_plot_acc, 
                    output_plot_meanacc, output_plot_loss
                ],
                queue=True  # 若训练耗时很长，建议这样
            )

            # 中止训练
            gr.Button("⏹️中止训练", variant='stop').click(stop_training)

    return app


# 模型验证
def app4():
    with gr.Blocks() as app:
        # 默认不展开介绍
        with gr.Accordion("工况识别模型性能评估模块(展开了解详细介绍)",open=False):
            gr.Markdown('''
    ##### -功能介绍
    抽油机井工况识别模型性能评估模块提供模型性能评估服务，用户可以利用系统默认数据集（或在示功图生成模块中生成的数据集）对系统已经训练完毕的模型（或在工况识别模型训练模块自行训练完毕的模型）进行模型性能评估，系统会自动进行模型评估测试并展示训练结果（模型总体性能表、模型各类别性能表、混淆矩阵图、ROC曲线图、PR曲线图和TSNE可视化图）。
    ##### 模型选择及参数
    1.用户可使用系统已经训练完毕的模型进行性能评估，也可以使用自己在工况识别模型训练模块训练到的模型进行性能评估，在模型选择下拉框中，模型名字+具体日期的选项为用户自行训练的模型，没有日期后缀的选项为系统原生模型；

    2.在性能评估参数设置方面，系统允许用户设置批次大小（batch_size），默认使用验证集进行模型性能评估，默认使用TSNE可视化对识别结果进行可视化分析，默认不使用F16半精度推理。

    3.上述参数设置完毕后，点击运行按钮，系统进入模型性能评估状态。

    ##### 训练结果展示

    1.在评估完毕后，系统会自动展示模型总体性能评估和模型各类别性能评估的表格；

    2.系统在模型性能评估完毕后，会展示模型混淆矩阵图、ROC曲线图、PR曲线图和TSNE可视化图，用户可放大浏览并下载。

    ''')
        with gr.Row():
            # 输入
            with gr.Column():
                with gr.Row():
                    save_path_in = gr.Dropdown(label="模型选择", choices= save_path_model(), value='repconvnest')
                with gr.Accordion("性能评估参数设置"):
                    with gr.Row():
                        task_in = gr.Radio(label="评估数据集选择（默认验证集）", choices=["测试集", "验证集","训练集"], value="验证集")
                        batch_size_in = gr.Number(label="批次大小（默认为64）",minimum=1, value=64)
                    with gr.Row():
                        tsne_in = gr.Radio(label="是否使用tsne可视化（默认使用）", choices=["是", "否"], value="是")
                        half_in = gr.Radio(label="是否使用F16半精度推理（默认不使用）", choices=["是", "否"], value="否")
            # 输出
            with gr.Column():
                with gr.Accordion("性能评估结果"):
                    output_all = gr.Dataframe(label='-模型总体性能评估-')
                    output_single = gr.Dataframe(label='-模型各类别性能评估-')
        with gr.Row():
            with gr.Accordion("性能评估可视化"):
                output_images = gr.Gallery(label="模型性能可视化", show_label=True,columns=[4], rows=[1], object_fit="contain", height="auto")

        with gr.Row():
            gr.Button("▶️运行",variant='primary').click(app4_fn, inputs=[save_path_in,task_in,batch_size_in,tsne_in,half_in],
                                    outputs=[output_all,output_single,output_images])
    return app

# 用户管理
def app5():
    with gr.Blocks(title="用户管理模块") as app:
        gr.Markdown("""
        ### 🧑‍💼 用户管理模块
        -具有管理员身份的账户可进行新账号注册和已有账号修改。
        """)

        with gr.Row():
            # 左侧：输入区域 + 提交按钮
            with gr.Column():
                with gr.Row():
                    login_account = gr.Textbox(
                        label="请输入您的登录账号", 
                        placeholder="输入登录账号"
                    )
                    login_password = gr.Textbox(
                        label="请输入您的登录密码", 
                        placeholder="输入登录密码", 
                        type="password"
                    )
                with gr.Row():
                    register_account = gr.Textbox(
                        label="请输入注册/已有账号（仅可输入大小写字母和数字）", 
                        placeholder="输入注册/已有账号"
                    )
                    register_password = gr.Textbox(
                        label="请输入注册密码/更改已有密码（仅可输入大小写字母和数字）", 
                        placeholder="输入注册密码/更改已有密码", 
                        type="password"
                    )
                role = gr.Dropdown(
                    choices=["管理员", "用户"], 
                    label="注册身份/更改已有身份"
                )
                submit_button = gr.Button("提交")

            # 右侧：输出区域
            with gr.Column():
                system_message = gr.Textbox(label="系统消息")
                user_list = gr.Dataframe(headers=["用户名", "身份"], label="🧑‍💼 用户列表")

        # 绑定事件
        submit_button.click(
            register,
            inputs=[login_account, login_password, register_account, register_password, role],
            outputs=[system_message, user_list]
        )

    return app




with gr.Blocks(title="抽油机井异常检测系统",theme = my_theme) as demo:
    with gr.Tab("功能介绍"):
        app()
    with gr.Tab("工控系统网络诊断"):
        app_ml()
    with gr.Tab("抽油机井工况诊断"):
        app1()
    with gr.Tab("示功图生成"):
        app2()
    # with gr.Tab("抽油机井工况识别模型训练"):
    #     app3()
    with gr.Tab("工况识别模型性能评估"):
        app4()
    with gr.Tab("用户管理"):
        app5()



def read_xlsx_to_list(file_path):
    # 读取xlsx文件
    df = pd.read_excel(file_path)
    # 提取第一列和第二列的数据
    col1 = df.iloc[:, 0]  # 第一列
    col2 = df.iloc[:, 1]  # 第二列
    # 将第一列和第二列的数据组合成元组，并放入列表中
    result = list(zip(col1, col2))
    return result

if __name__=='__main__':
    # 从用户数据文件中创建一个列表元组来存储用户信息
    users = read_xlsx_to_list("websystem_data/users.xlsx")
    # 启动Gradio服务器，share=True会生成一个72h的公共链接，到期则会自动关闭，再次launch会重新生成一个新的公共链接
    # inbrowser=True会在浏览器中自动打开Gradio界面
    demo.launch(auth=users,auth_message="请输入账号和密码",share=False,inbrowser=False)










