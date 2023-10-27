# 导入需要的库
import argparse # argparse库用于处理命令行参数
import os # os库用于处理操作系统相关的操作，如文件路径管理、环境变量管理等
import json # json库用于处理json格式的数据
import pprint # pprint库用于更美观的打印数据

# 导入transformers库中的相关模块
# AutoTokenizer用于自动地根据模型名称或路径加载对应的分词器
# AutoModelForSeq2SeqLM用于自动地根据模型名称或路径加载对应的序列到序列的语言模型
# TrainingArguments用于定义训练参数
# Trainer是一个用于模型训练和评估的类
# GenerationConfig用于定义生成（解码）过程中的配置，如最大长度、温度等
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, GenerationConfig

# 导入datasets库中的load_dataset函数，该函数用于加载数据集
from datasets import load_dataset

def list_files(startpath):
    """辅助函数，用于列出指定目录下的所有文件"""
    # 使用os库的walk函数遍历startpath目录及其所有子目录
    for root, dirs, files in os.walk(startpath):
        # 计算当前目录的深度，root.replace(startpath, '')会删除startpath部分，然后通过计算路径分隔符的数量得到深度
        level = root.replace(startpath, '').count(os.sep)
        # 根据目录深度计算缩进的空格数，每一级深度增加4个空格
        indent = ' ' * 4 * (level)
        # 输出当前目录的名称，basename函数会获取到路径的最后一部分即当前目录的名称
        print('{}{}/'.format(indent, os.path.basename(root)))
        # 计算子目录或文件的缩进空格数，比当前目录多4个空格
        subindent = ' ' * 4 * (level + 1)
        # 遍历当前目录下的所有文件，并输出
        for f in files:
            print('{}{}'.format(subindent, f))

def parse_args():
    # 创建一个ArgumentParser对象，它会保存所有必要的信息，以便以后从命令行解析数据。
    parser = argparse.ArgumentParser()

    # 使用add_argument方法添加参数，每种类型的参数都需要指定一个名称或者一个选项列表
    # type参数指定参数类型，default参数指定默认值
    # os.environ[]用于从环境变量中获取对应的值
    # 以下是各个参数的解释，具体含义根据实际情况可能有所不同
    parser.add_argument("--train_data", type=str, default=os.environ["SM_CHANNEL_TRAIN"])  # 训练数据
    parser.add_argument("--validation_data", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])  # 验证数据
    parser.add_argument("--test_data", type=str, default=os.environ["SM_CHANNEL_TEST"])  # 测试数据
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DIR"])  # 输出目录
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))  # 主机列表
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])  # 当前主机
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"])  # GPU数量
    parser.add_argument("--checkpoint_base_path", type=str, default="/opt/ml/checkpoints")  # 检查点基本路径
    parser.add_argument("--train_batch_size", type=int, default=128)  # 训练批次大小
    parser.add_argument("--validation_batch_size", type=int, default=256)  # 验证批次大小
    parser.add_argument("--test_batch_size", type=int, default=256)  # 测试批次大小
    parser.add_argument("--epochs", type=int, default=2)  # 训练的轮次
    parser.add_argument("--weight_decay", type=float, default=0.01)  # 权重衰减
    parser.add_argument("--learning_rate", type=float, default=0.00003)  # 学习率
    parser.add_argument("--train_sample_percentage", type=float, default=0.01)  # 训练样本百分比
    parser.add_argument("--model_checkpoint", type=str, default=None)  # 模型检查点    
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])  # 输出数据目录，此参数未使用

    # 使用parse_known_args方法解析命令行参数，这个方法会返回两个值，第一个是包含所有已知参数的对象，第二个是包含所有未知参数的列表
    args, _ = parser.parse_known_args()

    # 输出已知参数
    print("Args:")
    print(args)

    # 获取所有环境变量
    env_var = os.environ

    # 输出环境变量
    print("Environment Variables:")
    pprint.pprint(dict(env_var), width=1)

    # 返回解析得到的已知参数
    return args


if __name__ == "__main__":
    
    # 解析命令行参数
    args = parse_args()

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    # 探索输入文件
    local_data_processed_path = '/opt/ml/input/data'
    print('列出所有输入数据文件...')
    list_files(local_data_processed_path)

    # 加载数据集
    print(f'从以下路径加载数据集：{local_data_processed_path}')
    tokenized_dataset = load_dataset(
        local_data_processed_path,
        data_files={'train': 'train/*.parquet', 'test': 'test/*.parquet', 'validation': 'validation/*.parquet'}
    ).with_format("torch")
    print(f'加载的数据集：{tokenized_dataset}')
    
    # 对训练数据集进行采样
    skip_inds = int(1 / args.train_sample_percentage)
    sample_tokenized_dataset = tokenized_dataset.filter(lambda example, indice: indice % skip_inds == 0, with_indices=True)

    # 训练模型
    output_dir = args.checkpoint_base_path
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.validation_batch_size,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sample_tokenized_dataset['train'],
        eval_dataset=sample_tokenized_dataset['validation']
    )
    trainer.train()
    
    # 定义模型保存的路径
    transformer_fine_tuned_model_path = os.environ["SM_MODEL_DIR"]
    # 如果路径不存在则创建，如果目录已存在，则不会报错（exist_ok=True）
    os.makedirs(transformer_fine_tuned_model_path, exist_ok=True)
    # 打印保存模型的路径
    print(f"将最终模型保存到：transformer_fine_tuned_model_path={transformer_fine_tuned_model_path}")
    # 保存模型
    model.save_pretrained(transformer_fine_tuned_model_path)
    # 保存分词器
    tokenizer.save_pretrained(transformer_fine_tuned_model_path)
    
    # Copy inference.py and requirements.txt to the code/ directory for model inference
    #   Note: This is required for the SageMaker Endpoint to pick them up.
    #         This appears to be hard-coded and must be called code/
    
    # 当你在 Amazon SageMaker 中训练一个模型时，你的推理脚本（通常是一个 .py 文件）会被打包到生成的模型文件中。
    # 在训练结束后，SageMaker 会生成一个模型文件（通常是一个 .tar.gz 文件），并将其存储在 Amazon S3 中。这个模型文件包含了两部分内容：
    # 训练得到的模型参数，这是模型的核心部分，它定义了模型的行为。
    # code/ 文件夹，其中包含了你的推理脚本以及任何其他你在训练脚本中指定的辅助代码文件。
    # 当你使用这个模型文件创建一个 SageMaker 推理终端节点（Endpoint）或批量转换任务（Batch Transform Job）时，SageMaker 会从模型文件中解压出这些代码文件，并使用它们来处理推理请求。
    
    # 获取模型保存的路径
    # SM_MODEL_DIR是一个在Amazon SageMaker训练任务中自动配置的环境变量。它表示一个文件路径，这个路径用于存储训练过程中生成的模型文件。
    local_model_dir = os.environ["SM_MODEL_DIR"]
    # 指定推理代码的保存路径
    inference_path = os.path.join(local_model_dir, "code/")
    # 打印正在复制文件的信息
    print("正在将推理源文件复制到 {}".format(inference_path))
    # 创建推理代码的保存路径，如果路径已存在，则不会报错
    os.makedirs(inference_path, exist_ok=True)
    # 复制推理脚本到指定路径
    os.system("cp inference.py {}".format(inference_path))
    # 复制依赖文件到指定路径
    os.system('cp requirements.txt {}'.format(inference_path))
    # 打印推理代码路径中的文件列表
    print(f'在推理代码路径 "{inference_path}" 中的文件：')
    list_files(inference_path)
