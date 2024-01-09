import argparse
import os
import pathlib

from meta_critique.openai_config import OpenaiConfig
from meta_critique.utils import (
    build_message,
    generate_outputs,
    OpenAIChat,
    read_json,
    read_txt,
)


def eval_aiu_precision(
    batched_openai_engine,
    all_data,
    sys_msg_file="meta_critique/prompts/precision.txt",
    batch_size=5,
    cache_file="cache/eval_aiu_precision.json",
):
    sys_msg = read_txt(sys_msg_file)
    data_inputs = []
    for data in all_data:
        cur_input = (
            f"<input question>\n"
            f"{data['question'].strip()}\n\n"
            f"<model-generated answer>\n"
            f"{data['response'].strip()}\n\n"
            f"<reference answer>\n"
            f"{data['gpt4_answer'].strip()}\n\n"
        )

        aius = data["hypothesis_critique"]["aius"]
        for aiu in aius:
            usr_input = (
                cur_input + f"<claim>\n" f"{aiu.strip()}\n\n" f"<verify claim>\n"
            )
            data_inputs.append(build_message(sys_msg, usr_input))
    _, data_outputs = generate_outputs(
        data_inputs, batched_openai_engine, cache_file, batch_size, True
    )
    return data_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="The evaluation data with reference answer, "
        "reference critique and their aius in json format.",
    )

    parser.add_argument(
        "--out",
        default=None,
        type=str,
        required=True,
        help="The output file name.",
    )
    args = parser.parse_args()

    config = OpenaiConfig()
    batched_openai_engine = OpenAIChat(
        api_key=config.api_key,
        api_base=config.api_base,
        org_id=config.org_id,
        model=config.model_type,
        seed=config.seed,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        request_timeout=config.request_timeout,
    )

    all_data = read_json(args.data)
    prompts_path = os.path.join(os.path.dirname(pathlib.Path(__file__)), "prompts/")

    data_outputs = eval_aiu_precision(
        batched_openai_engine,
        all_data,
        sys_msg_file=os.path.join(prompts_path, "precision.txt"),
        batch_size=5,
        cache_file=args.out,
    )
