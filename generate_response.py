import json
import torch
from model_setup import (
    MODEL_ID,
    load_model,
    parse_question,
    prepare_input,
    print_section,
)


# 根据 thinking 控制 token，把模型输出拆成思考过程和正式回答
def split_response(tokenizer, response_ids):
    token_ids = response_ids.tolist()
    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    # 正常情况：以 </think> 为界，前面是思考，后面是正式回答
    if think_end_id in token_ids:
        split_at = token_ids.index(think_end_id)
        thinking_start = 1 if token_ids and token_ids[0] == think_start_id else 0
        thinking_ids = response_ids[thinking_start:split_at]
        answer_ids = response_ids[split_at + 1 :]
    # 可能出现长度截断：出现 <think>，但还没有生成 </think>
    elif token_ids and token_ids[0] == think_start_id:
        thinking_ids = response_ids[1:]
        answer_ids = response_ids[:0]
    # 没有 thinking 标记：全部 token 都是正式回答
    else:
        thinking_ids = response_ids[:0]
        answer_ids = response_ids

    # 最后把两段 token ID 分别还原成文字
    # 就得到了思维链和最终回复两部分内容
    return {
        "reasoning": tokenizer.decode(
            thinking_ids,
            skip_special_tokens=True,
        ).strip(),
        "content": tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        ).strip(),
    }


# 循环预测下一 token，再把完整输出整理成用户看到的响应
def generate_response(model, tokenizer, inputs):
    # 固定随机种子
    torch.manual_seed(42)

    # 获取输入的 token 长度
    input_length = inputs["input_ids"].shape[1]

    # generate 会重复执行“计算 logits -> 选择 token -> 追加到末尾”
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,

            # 生成token的上限
            max_new_tokens=256,
            do_sample=True,

            # 关键参数设置，参照hugging face 的 model card
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,

            # 开启 KV Cache
            use_cache=True,
        )

    # 切掉原始输入，只保留模型生成的部分
    response_ids = generated_ids[0, input_length:]
    raw_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    parsed = split_response(tokenizer, response_ids)

    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    finish_reason = "stop" if response_ids[-1].item() in eos_ids else "length"

    print_section("GENERATION")
    print(f"input token count    = {input_length}")
    print(f"full token count     = {generated_ids.shape[1]}")
    print(f"response token count = {response_ids.shape[0]}")
    print(f"response_ids         = {response_ids.tolist()}")
    print("raw_text:")
    print(raw_text)

    print_section("USER-FACING RESPONSE")
    print(
        json.dumps(
            {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning_content": parsed["reasoning"],
                            "content": parsed["content"],
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": response_ids.shape[0],
                    "total_tokens": generated_ids.shape[1],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if finish_reason == "stop" and not parsed["content"]:
        raise AssertionError("The model did not produce a final answer")


def main():
    question = parse_question()
    _, tokenizer, model, device = load_model()
    _, inputs = prepare_input(
        tokenizer,
        device,
        question,
        enable_thinking=True,
    )
    generate_response(model, tokenizer, inputs)


if __name__ == "__main__":
    main()
