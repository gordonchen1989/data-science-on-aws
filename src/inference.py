# 引入json模块，用于处理json数据
import json
# 引入logging模块，用于进行日志记录
import logging
# 引入typing模块的Any, Dict, Union，用于类型提示
from typing import Any
from typing import Dict
from typing import Union
# 引入subprocess模块，用于新开启子进程，并连接他们的输入/输出/错误管道，获取返回值等
import subprocess
# 引入sys模块，用于访问与Python解释器相关联的一些变量和函数
import sys

# 引入torch模块，是一个开源的机器学习库，提供了丰富的深度学习算法
import torch
# 引入transformers库的AutoTokenizer和AutoModelForSeq2SeqLM，用于处理自动化的文本标记化和序列到序列的语言模型
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# 引入GenerationConfig，用于配置生成模型的设置
from transformers import GenerationConfig

# 引入sagemaker_inference的encoder，这是一个用于Amazon SageMaker的推理编码器
from sagemaker_inference import encoder
# 引入TextGenerationPipeline，这是一个用于文本生成的管道
from transformers import TextGenerationPipeline
# 引入pipeline函数，用于创建处理模型输入和输出的流水线
from transformers import pipeline
# 引入set_seed函数，用于设置随机数生成器的种子，以确保实验的可复现性
from transformers import set_seed


# 定义一些常量，表示可能的HTTP内容类型
APPLICATION_X_TEXT = "application/x-text"
APPLICATION_JSON = "application/json"
# 定义字符串解码方式
STR_DECODE_CODE = "utf-8"

# 定义一个扩展常量，可能用于HTTP请求的内容类型，表示需要详细的响应
VERBOSE_EXTENSION = ";verbose"

# 定义一个常量，表示文本生成任务
TEXT_GENERATION = "text-generation"

# 定义生成的文本和文本数组的键
GENERATED_TEXT = "generated_text"
GENERATED_TEXTS = "generated_texts"

# 可能的模型参数常量
TEXT_INPUTS = "text_inputs"
MAX_LENGTH = "max_length"
NUM_RETURN_SEQUENCES = "num_return_sequences"
NUM_NEW_TOKENS = "num_new_tokens"
NUM_BEAMS = "num_beams"
TOP_P = "top_p"
EARLY_STOPPING = "early_stopping"
DO_SAMPLE = "do_sample"
NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
TOP_K = "top_k"
TEMPERATURE = "temperature"
SEED = "seed"

# 所有可能的参数名称的列表
ALL_PARAM_NAMES = [
    TEXT_INPUTS,
    MAX_LENGTH,
    NUM_NEW_TOKENS,
    NUM_RETURN_SEQUENCES,
    NUM_BEAMS,
    TOP_P,
    EARLY_STOPPING,
    DO_SAMPLE,
    NO_REPEAT_NGRAM_SIZE,
    TOP_K,
    TEMPERATURE,
    SEED,
]

# 定义模型参数的范围下限
# 定义最大长度的最小值
MAX_LENGTH_MIN = 1
# 定义返回序列数的最小值
NUM_RETURN_SEQUENCE_MIN = 1
# 定义波束搜索的最小波数
NUM_BEAMS_MIN = 1
# 定义Top-P采样的最小值
TOP_P_MIN = 0
# 定义Top-P采样的最大值
TOP_P_MAX = 1
# 定义不重复n-gram的最小大小
NO_REPEAT_NGRAM_SIZE_MIN = 1
# 定义Top-K采样的最小值
TOP_K_MIN = 0
# 定义温度的最小值
TEMPERATURE_MIN = 0




def model_fn(model_dir: str) -> list:
    """
    以模型委托人的身份创建我们的推理任务。

    这个函数在每个worker上只运行一次。

    Args:
        model_dir (str): 存储模型文件的目录
    Returns:
        list: 一个huggingface的tokenizer 和 model
    """
    
    # 打印正在遍历的模型目录
    print('walking model_dir: {}'.format(model_dir))

    # 引入os模块
    import os
    # 遍历模型目录，包括所有子目录和文件
    for root, dirs, files in os.walk(model_dir, topdown=False):
        for name in files:
            # 打印文件的完整路径
            print(os.path.join(root, name))
        for name in dirs:
            # 打印目录的完整路径
            print(os.path.join(root, name))
            
    # 加载tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    # 打印加载的本地HuggingFace tokenizer 
    print(f'Loaded Local HuggingFace Tokenzier:\n{tokenizer}')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    # 打印加载的本地HuggingFace模型
    print(f'Loaded Local HuggingFace Model:\n{model}')
    
    # 返回一个包含标记器和模型的列表
    return [tokenizer, model]


def _validate_payload(payload: Dict[str, Any]) -> None:
    """
    验证输入中的参数。

    检查max_length, num_return_sequences, num_beams, top_p和temperature是否在范围内。
    检查do_sample是否为布尔值。
    检查max_length, num_return_sequences, num_beams和seed是否为整数。

    Args:
        payload: 解码后的输入负载（输入参数和值的字典）
    """
    # 检查负载中的所有参数名称是否合法
    for param_name in payload:
        # assert (param_name in ALL_PARAM_NAMES) 这一行检查当前遍历到的 param_name 是否在 ALL_PARAM_NAMES 列表中。assert 语句用于测试后面的条件，如果条件为 False，它就会触发一个 AssertionError
        # f"Input payload contains an invalid key {param_name}. Valid keys are {ALL_PARAM_NAMES}." 这部分是 assert 语句的错误消息。如果 assert 的条件为 False，Python 就会抛出一个 AssertionError，并附带这个错误消息
        assert (
            param_name in ALL_PARAM_NAMES
        ), f"Input payload contains an invalid key {param_name}. Valid keys are {ALL_PARAM_NAMES}."

    # 检查负载中是否包含必要的参数TEXT_INPUTS
    assert TEXT_INPUTS in payload, f"Input payload must contain {TEXT_INPUTS} key."

    # 检查某些参数是否为整数
    for param_name in [MAX_LENGTH, NUM_RETURN_SEQUENCES, NUM_BEAMS, SEED]:
        if param_name in payload:
            assert type(payload[param_name]) == int, f"{param_name} must be an integer, got {payload[param_name]}."

    # 检查max_length是否在范围内
    if MAX_LENGTH in payload:
        assert (
            payload[MAX_LENGTH] >= MAX_LENGTH_MIN
        ), f"{MAX_LENGTH} must be at least {MAX_LENGTH_MIN}, got {payload[MAX_LENGTH]}."

    # 检查num_return_sequences是否在范围内
    if NUM_RETURN_SEQUENCES in payload:
        assert payload[NUM_RETURN_SEQUENCES] >= NUM_RETURN_SEQUENCE_MIN, (
            f"{NUM_RETURN_SEQUENCES} must be at least {NUM_RETURN_SEQUENCE_MIN}, "
            f"got {payload[NUM_RETURN_SEQUENCES]}."
        )

    # 检查num_beams是否在范围内
    if NUM_BEAMS in payload:
        assert (
            payload[NUM_BEAMS] >= NUM_BEAMS_MIN
        ), f"{NUM_BEAMS} must be at least {NUM_BEAMS_MIN}, got {payload[NUM_BEAMS]}."

    # 检查num_return_sequences是否小于或等于num_beams
    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS in payload:
        assert payload[NUM_RETURN_SEQUENCES] <= payload[NUM_BEAMS], (
            f"{NUM_BEAMS} must be at least {NUM_RETURN_SEQUENCES}. Instead got "
            f"{NUM_BEAMS}={payload[NUM_BEAMS]} and {NUM_RETURN_SEQUENCES}="
            f"{payload[NUM_RETURN_SEQUENCES]}."
        )

    # 检查top_p是否在范围内
    if TOP_P in payload:
        assert TOP_P_MIN <= payload[TOP_P] <= TOP_P_MAX, (
            f"{TOP_K} must be in range [{TOP_P_MIN},{TOP_P_MAX}], got "
            f"{payload[TOP_P]}"
        )

    # 检查temperature是否在范围内
    if TEMPERATURE in payload:
        assert payload[TEMPERATURE] >= TEMPERATURE_MIN, (
            f"{TEMPERATURE} must be a float with value at least {TEMPERATURE_MIN}, got "
            f"{payload[TEMPERATURE]}."
        )

    # 检查do_sample是否为布尔值
    if DO_SAMPLE in payload:
        assert (
            type(payload[DO_SAMPLE]) == bool
        ), f"{DO_SAMPLE} must be a boolean, got {payload[DO_SAMPLE]}."


def _update_num_beams(payload: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, int]]:
    """
    如果负载中缺少num_beams但存在num_return_sequences，则向负载中添加num_beams。

    Args:
        payload (dict): 输入的参数字典
    Returns:
        dict: 更新后的参数字典
    """
    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS not in payload:
        # 如果payload中包含num_return_sequences但不包含num_beams，则添加num_beams
        payload[NUM_BEAMS] = payload[NUM_RETURN_SEQUENCES]
    return payload


def transform_fn(model_objs: list, input_data: bytes, content_type: str, accept: str) -> bytes:
    """
    对模型进行预测并返回序列化的响应。

    函数签名符合SM contract。

    Args:
        model_objs (list): tokenizer，模型
        input_data (obj): 请求数据。
        content_type (str): 请求内容类型。
        accept (str): 客户端期望的接受头部。
    Returns:
        obj: 预测的字节字符串
    """
    # 从模型对象中获取分词器和模型
    tokenizer = model_objs[0]
    model = model_objs[1]
    
    # 如果内容类型是文本，解码输入数据
    if content_type == APPLICATION_X_TEXT:
        try:
            # STR_DECODE_CODE="utf-8"
            input_text = input_data.decode(STR_DECODE_CODE)
        except Exception:
            logging.exception(
                f"Failed to parse input payload. For content_type={APPLICATION_X_TEXT}, input "
                f"payload must be a string encoded in utf-8 format."
            )
            raise
        try:
            # 将输入数据传递给模型进行预测，并解码预测结果
            # 使用分词器（tokenizer）将输入文本（input_text）转化为模型可以理解的输入形式，具体来说，它将文本转换为一个包含词汇表索引的张量（tensor）。这里的return_tensors="pt"是指示分词器返回PyTorch张量（'pt'代表PyTorch）。
            # input_ids是一个整数列表，其中每个整数代表词汇表中词的索引。这个列表表示了输入文本中的每个词或子词。模型（如BERT、GPT-2等）将使用这个列表作为输入，进行接下来的预测或分类等任务。
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            # 使用模型对象（model）的 generate 方法生成新的文本序列。这个方法接收很多参数，可以用来指定生成过程的很多方面。
            # 函数的输入是 input_ids，它是一个由分词器生成的 token ID 列表，表示已知的输入序列。模型将从这个输入序列开始生成新的 token。
            # max_length 参数指定生成的文本序列的最大长度。如果模型在达到这个长度之前没有生成一个结束符号（如句号或问号），它将停止生成更多的 token。
            original_outputs = model.generate(input_ids,
                                              GenerationConfig(max_new_tokens=200)
                                             )
            # 使用 tokenizer 对原始输出进行解码
            # 'original_outputs[0]' 是我们要解码的内容
            # 'skip_special_tokens=True' 表示在解码时跳过特殊的令牌（例如，填充或分隔符等）
            output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
        except Exception:
            logging.exception("Failed to do inference")
            raise

    # TODO: Incorporate JSON implementation for more inference options here
    # elif content_type == APPLICATION_JSON:
    #     try:
    #         payload = json.loads(input_data)
    #     except Exception:
    #         logging.exception(
    #             f"Failed to parse input payload. For content_type={APPLICATION_JSON}, input "
    #             f"payload must be a json encoded dictionary with keys {ALL_PARAM_NAMES}."
    #         )
    #         raise
    #     _validate_payload(payload)
    #     payload = _update_num_beams(payload)
    #     if SEED in payload:
    #         set_seed(payload[SEED])
    #         del payload[SEED]
    #     try:
    #         model_output = text_generator(**payload)
    #         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    #         original_outputs = model.generate(input_ids,
    #                                           GenerationConfig(max_new_tokens=200)
    #                                          )
    #         model_output = tokenizer.decode(original_outputs[0], skip_special_tokens=True)
    #         output = {GENERATED_TEXTS: [x[GENERATED_TEXT] for x in model_output]}
    #     except Exception:
    #         logging.exception("Failed to do inference")
    #         raise
      
    else:
        raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
    # # 如果 'accept' 的后缀是 'VERBOSE_EXTENSION'，则进行处理
    if accept.endswith(VERBOSE_EXTENSION):
        # 使用 'rstrip' 移除 'accept' 字符串末尾的 'VERBOSE_EXTENSION'
        # 'VERBOSE_EXTENSION' 是详细扩展名，例如可以是 ';verbose' 这样的字符串
        accept = accept.rstrip(VERBOSE_EXTENSION)  # Verbose and non-verbose response are identical
    # 使用 'encoder' 对 'output' 进行编码，并指定 'accept' 作为编码类型
    # 'output' 是需要编码的内容，'accept' 是编码类型，例如可以是 'json'、'text' 等
    # 编码后的结果返回
    return encoder.encode(output, accept)
