# 导入 subprocess 模块，该模块可以用来创建新的进程，并连接到其输入/输出/错误管道，获取返回值等
import subprocess

# 导入 sys 模块，该模块提供对 Python 解释器使用或维护的一些变量的访问
import sys

# 导入 json 模块，该模块提供了 JSON 数据解析的方法
import json

# 导入 argparse 模块，该模块提供了创建命令行参数和选项的方法
import argparse

# 使用 subprocess 模块的 check_call 方法运行 pip 安装命令，安装特定版本的 transformers、datasets 和 torch 库
# sys.executable 是 Python 解释器的路径，"-m" 是一个命令行选项，表示后面的字符串应作为模块名来执行
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.26.1", "datasets==2.9.0", "torch==1.13.1"])

# 从 transformers 库中导入 AutoTokenizer 类，该类可以自动加载预训练的分词器
from transformers import AutoTokenizer

# 从 datasets 库中导入 load_dataset 函数和 DatasetDict 类，load_dataset 函数可以加载各种 NLP 数据集，DatasetDict 类可以管理多个数据集
from datasets import load_dataset, DatasetDict

# 导入 os 模块，该模块提供了大量的函数来处理文件和目录
import os

# 导入 time 模块，该模块提供了时间相关的函数
import time


# 定义一个 transform_dataset 函数，参数包括输入数据的路径、输出数据的路径、预训练模型名，以及训练集、测试集和验证集的划分比率
def transform_dataset(input_data,
                      output_data,
                      huggingface_model_name,
                      train_split_percentage,
                      test_split_percentage,
                      validation_split_percentage,
                      ):

    # 加载原始数据集
    dataset = load_dataset(input_data)
    print(f'数据集已从路径 {input_data} 加载\n{dataset}')
    
    # 加载分词器
    print(f'正在加载模型 {huggingface_model_name} 的分词器')
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
    
    # 对数据集进行训练集、测试集和验证集的划分
    train_testvalid = dataset['train'].train_test_split(1 - train_split_percentage, seed=1234)
    test_valid = train_testvalid['test'].train_test_split(test_split_percentage / (validation_split_percentage + test_split_percentage), seed=1234)
    # 创建 DatasetDict 对象，它是 Hugging Face 的 datasets 库中的一个类。DatasetDict 是一个字典，它的键是字符串（如 "train"、"test"、"validation"），值是 Dataset 对象。这样，通过 DatasetDict，可以轻松地管理训练、测试和验证数据集，并能方便地对这些数据集进行操作，如应用相同的预处理函数等。
    train_test_valid_dataset = DatasetDict(
        {
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']
        }
    )
    print(f'数据集划分后的情况:\n{train_test_valid_dataset}')
    
    # 创建一个标记化函数
    # 定义一个 tokenize_function 函数，参数是一个样本
    def tokenize_function(example):
        # 定义提示信息
        prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '

        # 将提示信息添加到对话内容的前后，形成新的输入内容
        inp = [prompt + i + end_prompt for i in example["dialogue"]]

        # 使用分词器对新的输入内容进行分词，返回的是一个输入ID列表，将其添加到样本字典的 'input_ids' 键中
        # padding="max_length" 表示将分词结果填充到最大长度，truncation=True 表示如果分词结果超过最大长度则截断，return_tensors="pt" 表示返回 PyTorch 张量
        example['input_ids'] = tokenizer(inp, padding="max_length", truncation=True, return_tensors="pt").input_ids

        # 使用分词器对样本的 'summary' 字段进行分词，返回的是一个标签ID列表，将其添加到样本字典的 'labels' 键中
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

        # 返回处理过的样本
        return example
    
    # 对数据集进行标记化
    print(f'正在对数据集进行标记化...')
    # 使用 Dataset.map 方法对每个样本应用 tokenize_function 函数
    # batched=True 表示以批次方式执行该操作，这样可以加快处理速度
    tokenized_datasets = train_test_valid_dataset.map(tokenize_function, batched=True)

    # 使用 Dataset.remove_columns 方法删除 'id', 'topic', 'dialogue', 'summary' 这四列
    # 因为这些列在之后的模型训练中不再需要
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
    print(f'标记化完成！')
    
    # 创建保存数据的目录
    os.makedirs(f'{output_data}/train/', exist_ok=True)
    os.makedirs(f'{output_data}/test/', exist_ok=True)
    os.makedirs(f'{output_data}/validation/', exist_ok=True)
    file_root = str(int(time.time()*1000))
    
    # 将数据集保存到磁盘
    print(f'正在将数据集写入到 {output_data}')
    tokenized_datasets['train'].to_parquet(f'./{output_data}/train/{file_root}.parquet')
    tokenized_datasets['test'].to_parquet(f'./{output_data}/test/{file_root}.parquet')
    tokenized_datasets['validation'].to_parquet(f'./{output_data}/validation/{file_root}.parquet')
    print('预处理完成！')

    
# 定义一个 process 函数，参数是一个包含各种参数的对象
def process(args):
    # 打印输入数据的路径，并列出该路径下的所有文件
    print(f"Listing contents of {args.input_data}")
    dirs_input = os.listdir(args.input_data)
    for file in dirs_input:
        print(file)

    # 调用 transform_dataset 函数处理数据
    # 输入数据的路径、输出数据的路径、预训练模型名，以及训练集、测试集和验证集的划分比率均从 args 对象中获取
    transform_dataset(input_data=args.input_data,
                      output_data=args.output_data,
                      huggingface_model_name=args.model_checkpoint,
                      train_split_percentage=args.train_split_percentage,
                      test_split_percentage=args.test_split_percentage,
                      validation_split_percentage=args.validation_split_percentage
                     )

    # 打印输出数据的路径，并列出该路径下的所有文件
    print(f"Listing contents of {args.output_data}")
    dirs_output = os.listdir(args.output_data)
    for file in dirs_output:
        print(file)

        
# 定义一个 list_arg 函数，参数是一个字符串
# 这个函数被用作 argparse 的自定义类型，用于解析一个字符串列表参数
def list_arg(raw_value):
    # 使用字符串的 split 方法以逗号为分隔符将字符串分割成一个列表，并返回该列表
    return str(raw_value).split(",")
        
        
def parse_args():
    # 首先尝试从 "/opt/ml/config/resourceconfig.json" 文件中读取资源配置信息
    # 如果该文件不存在，那么将打印一条错误消息并忽略该错误
    resconfig = {}
    try:
        with open("/opt/ml/config/resourceconfig.json", "r") as cfgfile:
            resconfig = json.load(cfgfile)
    except FileNotFoundError:
        print("/opt/ml/config/resourceconfig.json not found.  current_host is unknown.")
        pass  # Ignore

    # 创建一个 ArgumentParser 对象
    # argparse.ArgumentParser 是 Python 内置库 argparse 中的一个类，它被用于创建命令行参数和选项解析器。
    parser = argparse.ArgumentParser(description="Process")
    
    # 添加各种参数
    # 注意其中一些参数的默认值是从 resconfig 字典中获取的
    # 另外注意 `--hosts` 参数的类型是 list_arg，这是一个自定义的类型，用于解析一个字符串列表参数
    parser.add_argument(
        "--hosts",
        type=list_arg,
        default=resconfig.get("hosts", ["unknown"]),
        help="Comma-separated list of host names running the job",
    )
    parser.add_argument(
        "--current-host",
        type=str,
        default=resconfig.get("current_host", "unknown"),
        help="Name of this host running the job",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/data",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="/opt/ml/processing/output/data",
    )
    parser.add_argument(
        "--train-split-percentage",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--validation-split-percentage",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--test-split-percentage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="google/flan-t5-base"
    )
    # parser.add_argument(
    #     "--dataset-templates-name",
    #     type=str,
    #     default="amazon_us_reviews/Wireless_v1_00",
    # )
    # parser.add_argument(
    #     "--prompt-template-name",
    #     type=str,
    #     default="Given the review body return a categorical rating",
    # )
    
    # 使用 ArgumentParser 的 parse_args 方法解析命令行参数并返回结果
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Loaded arguments:")
    print(args)

    print("Environment variables:")
    print(os.environ)

    process(args)
