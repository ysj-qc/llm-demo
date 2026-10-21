import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


# 读取配置、Tokenizer 和权重文件，把完整模型加载进内存
def load_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16 if device.type != "cpu" else torch.float32

    # 读取 config.json，得到层数、隐藏维度、注意力头数等结构参数
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # 读取 tokenizer.json 和 tokenizer_config.json
    # 得到文字与 token ID 的转换规则，以及模型自带的 Chat Template
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # 根据 config 创建 28 层神经网络，并从 model.safetensors 加载模型权重
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        torch_dtype=dtype,
    )

    model.to(device)
    model.eval()
    return config, tokenizer, model, device


# 调用对话模板，再把结果转换成 PyTorch token 张量
def prepare_input(tokenizer, device, question, enable_thinking):
    # 把命令行问题组织成应用程序常用的 messages 格式
    messages = [
        {
            "role": "user", 
            "content": question,
        }
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    return prompt, inputs


def parse_question():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    return parser.parse_args().question


def print_section(title):
    print(f"\n=== {title} ===")
