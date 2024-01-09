import argparse

from meta_critique.utils import read_json, write_json


def merge_outcomes(
    all_data, ref_answer, ref_critique, ref_aius, hyp_critique, hyp_aius
):
    outputs = []
    for data_idx, data in enumerate(all_data):
        if ref_critique is not None:
            if "gpt4_critique" in data:
                data["gpt4_critique"]["critique"] = ref_critique[data_idx]["output"]
            else:
                data["gpt4_critique"] = {"critique": ref_critique[data_idx]["output"]}

        if ref_aius is not None:
            data["gpt4_critique"]["aius"] = (
                ref_aius[data_idx]["output"].strip().split("\n")
            )

        if ref_answer is not None:
            data["gpt4_answer"] = ref_answer[data_idx]["output"]

        data["hypothesis_critique"] = {
            "critique": hyp_critique[data_idx]["output"],
            "aius": hyp_aius[data_idx]["output"].strip().split("\n"),
        }
        outputs.append(data)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="The benchmark data in json format.",
    )
    parser.add_argument(
        "--ref_answer",
        default=None,
        type=str,
        help="The reference answer file.",
    )
    parser.add_argument(
        "--ref_critique",
        default=None,
        type=str,
        help="The reference critique file.",
    )
    parser.add_argument(
        "--ref_aius",
        default=None,
        type=str,
        help="The aius of reference aius file name.",
    )
    parser.add_argument(
        "--hyp_critique",
        default=None,
        type=str,
        required=True,
        help="The aius of hypothesis aius  file name.",
    )
    parser.add_argument(
        "--hyp_aius",
        default=None,
        type=str,
        required=True,
        help="The aius of hypothesis aius  file name.",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        required=True,
        help="The output file name.",
    )
    args = parser.parse_args()

    all_data = read_json(args.data)
    if args.ref_answer is not None:
        ref_answer = read_json(args.ref_answer)
    else:
        ref_answer = None

    if args.ref_critique is not None:
        ref_critique = read_json(args.ref_critique)
    else:
        ref_critique = None

    if args.ref_aius is not None:
        ref_aius = read_json(args.ref_aius)
    else:
        ref_aius = None

    hyp_critique = read_json(args.hyp_critique)
    hyp_aius = read_json(args.hyp_aius)

    data_outputs = merge_outcomes(
        all_data, ref_answer, ref_critique, ref_aius, hyp_critique, hyp_aius
    )

    write_json(data_outputs, args.out)
