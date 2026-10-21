import json

import torch

from model_setup import load_model, parse_question, prepare_input, print_section


# 打印模型层数、维度和注意力头等关键结构
def inspect_model(config, model, device):
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    print_section("MODEL")
    print(
        json.dumps(
            {
                "model_class": model.__class__.__name__,
                "device": device.type,
                "weight_dtype": str(next(model.parameters()).dtype),
                "total_parameters": total_parameters,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": len(model.base_model.layers),
                "num_attention_heads": config.num_attention_heads,
            },
            indent=2,
        )
    )

    # 打印第 0 层 Transformer 的真实结构
    print_section("LAYER 0")
    print(model.base_model.layers[0])


# 打印对话模板生成的文本，以及 Tokenizer 转换出的 token ID
def inspect_input(tokenizer, prompt, inputs, enable_thinking):
    print_section("CHAT TEMPLATE")
    print(f"enable_thinking = {enable_thinking}")
    print(prompt.replace("\n", "\\n\n"))

    # 打印 token ID 的详细情况
    print_section("TOKENS")
    input_ids = inputs["input_ids"][0]
    for index, token_id in enumerate(input_ids.tolist()):
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"{index:02d}  id={token_id:6d}  text={piece!r}")
    print(f"input_ids.shape      = {list(inputs['input_ids'].shape)}")


# 预测下一个 token，同时打印中间向量和原始分数
def inspect_forward(model, tokenizer, inputs):
    input_ids = inputs["input_ids"]

    # 在 embedding 权重矩阵中查表，把每个 token ID 换成 1024 维向量
    embeddings = model.get_input_embeddings()(input_ids)

    print_section("EMBEDDINGS")
    print(f"embeddings.shape        = {list(embeddings.shape)}")

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            # 单步计算关闭KV Cache
            use_cache=False,
        )

    if len(outputs.hidden_states) != model.config.num_hidden_layers + 1:
        raise AssertionError("Unexpected hidden state count")

    print_section("HIDDEN STATES")
    print(f"hidden_states.count     = {len(outputs.hidden_states)}")
    print(f"hidden_states[0].shape  = {list(outputs.hidden_states[0].shape)}")
    print(f"hidden_states[-1].shape = {list(outputs.hidden_states[-1].shape)}")
    first_values = outputs.hidden_states[0][0, -1, :6].float().cpu().tolist()
    last_values = outputs.hidden_states[-1][0, -1, :6].float().cpu().tolist()
    print(f"last token before layer 0  = {[round(value, 5) for value in first_values]}")
    print(f"last token final state     = {[round(value, 5) for value in last_values]}")

    print_section("NEXT TOKEN LOGITS")
    print(f"logits.shape = {list(outputs.logits.shape)}")

    probabilities = torch.softmax(outputs.logits[0, -1].float(), dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=5)
    for token_id, probability in zip(top_ids.tolist(), top_probabilities.tolist()):
        token = tokenizer.decode([token_id], skip_special_tokens=False)
        logit = outputs.logits[0, -1, token_id].float().item()
        print(
            f"id={token_id:6d}  token={token!r:14}  "
            f"logit={logit:.4f}  probability={probability:.6f}"
        )


def main():
    question = parse_question()
    config, tokenizer, model, device = load_model()
    enable_thinking = True
    prompt, inputs = prepare_input(
        tokenizer,
        device,
        question,
        enable_thinking=enable_thinking,
    )

    inspect_model(config, model, device)
    inspect_input(tokenizer, prompt, inputs, enable_thinking)
    inspect_forward(model, tokenizer, inputs)


if __name__ == "__main__":
    main()
