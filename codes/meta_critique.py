import os.path

from openai_config import OpenaiConfig
from utils import read_json, write_json, OpenAIChat
import argparse
from generate_ref_answer import generate_ref_answer
from generate_ref_critique import generate_ref_critique
from extracting_aius_for_critique import extract_aius_for_critique
from merge_files import merge_outcomes
from evaluate_aiu_precision import eval_aiu_precision
from evaluate_aiu_recall import eval_aiu_recall
from cal_meta_scores import meta_score


def add_args(parser):
    parser.add_argument(
        "--data_w_o_reference",
        default=None,
        type=str,
        help="The data without reference answer and critique in json format",
    )

    parser.add_argument(
        "--benchmark_data",
        default=None,
        type=str,
        help="The data with reference answer and critique in json format",
    )

    parser.add_argument(
        "--cache_dir",
        default="tmp_cache/",
        type=str,
        help="The cache directory to save process results.",
    )

    parser.add_argument(
        "--hyp_critique",
        default=None,
        type=str,
        required=True,
        help="The hypothesis critique in json format.",
    )

    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="The output file name.",
    )

    return parser


def evaluate(args, batched_openai_engine):
    precision_score, recall_score, f1_score = 0, 0, 0
    hyp_critique = read_json(args.hyp_critique)
    hyp_critique_list = [i["output"] for i in hyp_critique]
    print("extracting aius from hypothesis critique ...")
    hyp_aius = extract_aius_for_critique(batched_openai_engine, hyp_critique_list, sys_msg_file="prompts/extract_aius.txt", batch_size=5, cache_file=os.path.join(args.cache_dir, "hyp_aius.json"))

    if args.benchmark_data is not None:
        benchmark_data = read_json(args.benchmark_data)
        all_data = merge_outcomes(benchmark_data, None, None, None, hyp_critique, hyp_aius)
    elif args.data_w_o_reference is not None:
        question_data = read_json(args.data_w_o_reference)
        print("generating reference answer ...")
        ref_answer = generate_ref_answer(batched_openai_engine, question_data, sys_msg_file="prompts/reference_answer.txt", batch_size=5, cache_file=os.path.join(args.cache_dir, "ref_answer.json"))
        print("generating reference critique ...")
        ref_critique = generate_ref_critique(batched_openai_engine, question_data, sys_msg_file="prompts/reference_critique.txt", batch_size=5, cache_file=os.path.join(args.cache_dir, "ref_critique.json"))
        ref_critique_list = [i["output"] for i in ref_critique]
        print("extracting aius from reference critique ...")
        ref_aius = extract_aius_for_critique(batched_openai_engine, ref_critique_list, sys_msg_file="prompts/extract_aius.txt", batch_size=5, cache_file=os.path.join(args.cache_dir, "ref_aius.json"))
        all_data = merge_outcomes(question_data, ref_answer, ref_critique, ref_aius, hyp_critique, hyp_aius)
    else:
        print("You should input benchmark_data or data_w_o_reference!!!")
        return None, None, None
    print("performing precision task ...")
    precision_outputs = eval_aiu_precision(batched_openai_engine, all_data, sys_msg_file="prompts/precision.txt", batch_size=5, cache_file=os.path.join(args.cache_dir, "hypothesis_precision.json"))
    print("performing recall task ...")
    recall_outputs = eval_aiu_recall(batched_openai_engine, all_data, sys_msg_file="prompts/recall.txt", batch_size=5, cache_file=os.path.join(args.cache_dir, "hypothesis_recall.json"))

    precision_score, recall_score, f1_score = meta_score(all_data, precision_outputs, recall_outputs)
    return precision_score, recall_score, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    config = OpenaiConfig()
    batched_openai_engine = OpenAIChat(api_key=config.api_key, api_base=config.api_base, org_id=config.org_id, model=config.model_type, seed=config.seed, temperature=config.temperature, max_tokens=config.max_tokens, top_p=config.top_p, frequency_penalty=config.frequency_penalty, presence_penalty=config.presence_penalty, request_timeout=config.request_timeout)

    precision_score, recall_score, f1_score = evaluate(args, batched_openai_engine)

    print("Meta-P:", precision_score)
    print("Meta-R:", recall_score)
    print("Meta-F1:", f1_score)

    write_json({"precision": precision_score, "recall": recall_score, "f1_score": f1_score}, args.out)
