import argparse
import os
import pathlib

from meta_critique.cal_meta_scores import meta_score
from meta_critique.evaluate_aiu_precision import eval_aiu_precision
from meta_critique.evaluate_aiu_recall import eval_aiu_recall
from meta_critique.extracting_aius_for_critique import extract_aius_for_critique
from meta_critique.generate_ref_answer import generate_ref_answer
from meta_critique.generate_ref_critique import generate_ref_critique
from meta_critique.merge_files import merge_outcomes
from meta_critique.openai_config import OpenaiConfig
from meta_critique.utils import OpenAIChat, read_json, write_json


class MetaCritique:
    def __init__(
        self,
        model_type="gpt-4",
        batch_size=5,
        api_key=None,
        api_base=None,
        seed=None,
        cache_dir="tmp_cache",
    ):
        cur_config = OpenaiConfig()
        cur_config.model_type = model_type
        cur_config.seed = seed
        cur_config.api_key = api_key
        cur_config.api_base = api_base
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.batched_openai_engine = OpenAIChat(
            api_key=cur_config.api_key,
            api_base=cur_config.api_base,
            org_id=cur_config.org_id,
            model=cur_config.model_type,
            seed=cur_config.seed,
            temperature=cur_config.temperature,
            max_tokens=cur_config.max_tokens,
            top_p=cur_config.top_p,
            frequency_penalty=cur_config.frequency_penalty,
            presence_penalty=cur_config.presence_penalty,
            request_timeout=cur_config.request_timeout,
        )
        self.prompts_path = os.path.join(
            os.path.dirname(pathlib.Path(__file__)), "prompts/"
        )

    def score(self, input_list):
        assert len(input_list) > 0

        if "reference_answer" in input_list[0]:
            for tmp_data in input_list:
                tmp_data["gpt4_answer"] = tmp_data["reference_answer"]
                del tmp_data["reference_answer"]
        else:
            print("generating reference answer ...")
            ref_answers = generate_ref_answer(
                self.batched_openai_engine,
                input_list,
                sys_msg_file=os.path.join(self.prompts_path, "reference_answer.txt"),
                batch_size=self.batch_size,
                cache_file=os.path.join(self.cache_dir, "ref_answer.json"),
            )
            for idx, ref_answer in enumerate(ref_answers):
                tmp_data = input_list[idx]
                tmp_data["gpt4_answer"] = ref_answer["output"]

        if "reference_critique" in input_list[0]:
            for tmp_data in input_list:
                if isinstance(tmp_data["reference_critique"], dict):
                    assert "aius" in tmp_data["reference_critique"]
                    tmp_data["gpt4_critique"] = tmp_data["reference_critique"]
                else:
                    tmp_data["gpt4_critique"] = {
                        "critique": tmp_data["reference_critique"]
                    }
                del tmp_data["reference_critique"]
        else:
            print("generating reference critique ...")
            ref_critiques = generate_ref_critique(
                self.batched_openai_engine,
                input_list,
                sys_msg_file=os.path.join(self.prompts_path, "reference_critique.txt"),
                batch_size=self.batch_size,
                cache_file=os.path.join(self.cache_dir, "ref_critique.json"),
            )
            for idx, ref_critique in enumerate(ref_critiques):
                tmp_data = input_list[idx]
                tmp_data["gpt4_critique"] = {"critique": ref_critique["output"]}

        for tmp_data in input_list:
            tmp_data["hypothesis_critique"] = {
                "critique": tmp_data["hypothesis_critique"]
            }

        hyp_critique_list = [i["hypothesis_critique"]["critique"] for i in input_list]

        if "aius" not in input_list[0]["gpt4_critique"]:
            ref_critique_list = [i["gpt4_critique"]["critique"] for i in input_list]
            print("extracting aius from reference critique ...")
            ref_aius = extract_aius_for_critique(
                self.batched_openai_engine,
                ref_critique_list,
                sys_msg_file=os.path.join(self.prompts_path, "extract_aius.txt"),
                batch_size=self.batch_size,
                cache_file=os.path.join(self.cache_dir, "ref_aius.json"),
            )
            for idx, ref_aiu in enumerate(ref_aius):
                tmp_data = input_list[idx]
                tmp_data["gpt4_critique"]["aius"] = (
                    ref_aiu["output"].strip().split("\n")
                )

        print("extracting aius from hypothesis critique ...")
        hyp_aius = extract_aius_for_critique(
            self.batched_openai_engine,
            hyp_critique_list,
            sys_msg_file=os.path.join(self.prompts_path, "extract_aius.txt"),
            batch_size=self.batch_size,
            cache_file=os.path.join(self.cache_dir, "hyp_aius.json"),
        )
        for idx, hyp_aiu in enumerate(hyp_aius):
            tmp_data = input_list[idx]
            tmp_data["hypothesis_critique"]["aius"] = (
                hyp_aiu["output"].strip().split("\n")
            )

        print("performing precision task ...")
        precision_outputs = eval_aiu_precision(
            self.batched_openai_engine,
            input_list,
            sys_msg_file=os.path.join(self.prompts_path, "precision.txt"),
            batch_size=self.batch_size,
            cache_file=os.path.join(self.cache_dir, "hypothesis_precision.json"),
        )

        print("performing recall task ...")
        recall_outputs = eval_aiu_recall(
            self.batched_openai_engine,
            input_list,
            sys_msg_file=os.path.join(self.prompts_path, "recall.txt"),
            batch_size=self.batch_size,
            cache_file=os.path.join(self.cache_dir, "hypothesis_recall.json"),
        )

        precision_score, recall_score, f1_score = meta_score(
            input_list, precision_outputs, recall_outputs
        )
        for tmp_data in input_list:
            tmp_data["reference_answer"] = tmp_data["gpt4_answer"]
            tmp_data["reference_critique"] = tmp_data["gpt4_critique"]
            del tmp_data["gpt4_answer"]
            del tmp_data["gpt4_critique"]
        return precision_score, recall_score, f1_score


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
    prompts_path = os.path.join(os.path.dirname(pathlib.Path(__file__)), "prompts/")

    hyp_critique = read_json(args.hyp_critique)
    hyp_critique_list = [i["output"] for i in hyp_critique]
    print("extracting aius from hypothesis critique ...")
    hyp_aius = extract_aius_for_critique(
        batched_openai_engine,
        hyp_critique_list,
        sys_msg_file=os.path.join(prompts_path, "extract_aius.txt"),
        batch_size=5,
        cache_file=os.path.join(args.cache_dir, "hyp_aius.json"),
    )

    if args.benchmark_data is not None:
        benchmark_data = read_json(args.benchmark_data)
        all_data = merge_outcomes(
            benchmark_data, None, None, None, hyp_critique, hyp_aius
        )
    elif args.data_w_o_reference is not None:
        question_data = read_json(args.data_w_o_reference)
        print("generating reference answer ...")
        ref_answer = generate_ref_answer(
            batched_openai_engine,
            question_data,
            sys_msg_file=os.path.join(prompts_path, "reference_answer.txt"),
            batch_size=5,
            cache_file=os.path.join(args.cache_dir, "ref_answer.json"),
        )
        print("generating reference critique ...")
        ref_critique = generate_ref_critique(
            batched_openai_engine,
            question_data,
            sys_msg_file=os.path.join(prompts_path, "reference_critique.txt"),
            batch_size=5,
            cache_file=os.path.join(args.cache_dir, "ref_critique.json"),
        )
        ref_critique_list = [i["output"] for i in ref_critique]
        print("extracting aius from reference critique ...")
        ref_aius = extract_aius_for_critique(
            batched_openai_engine,
            ref_critique_list,
            sys_msg_file=os.path.join(prompts_path, "extract_aius.txt"),
            batch_size=5,
            cache_file=os.path.join(args.cache_dir, "ref_aius.json"),
        )
        all_data = merge_outcomes(
            question_data, ref_answer, ref_critique, ref_aius, hyp_critique, hyp_aius
        )
    else:
        print("You should input benchmark_data or data_w_o_reference!!!")
        return None, None, None
    print("performing precision task ...")
    precision_outputs = eval_aiu_precision(
        batched_openai_engine,
        all_data,
        sys_msg_file=os.path.join(prompts_path, "precision.txt"),
        batch_size=5,
        cache_file=os.path.join(args.cache_dir, "hypothesis_precision.json"),
    )
    print("performing recall task ...")
    recall_outputs = eval_aiu_recall(
        batched_openai_engine,
        all_data,
        sys_msg_file=os.path.join(prompts_path, "recall.txt"),
        batch_size=5,
        cache_file=os.path.join(args.cache_dir, "hypothesis_recall.json"),
    )

    precision_score, recall_score, f1_score = meta_score(
        all_data, precision_outputs, recall_outputs
    )
    return precision_score, recall_score, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
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

    precision_score, recall_score, f1_score = evaluate(args, batched_openai_engine)

    print("Meta-P:", precision_score)
    print("Meta-R:", recall_score)
    print("Meta-F1:", f1_score)

    write_json(
        {"precision": precision_score, "recall": recall_score, "f1_score": f1_score},
        args.out,
    )
