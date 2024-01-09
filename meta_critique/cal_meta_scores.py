import argparse

from meta_critique.utils import read_json, write_json


def f1_score(p, r):
    if p == 0 or r == 0:
        return 0
    else:
        return 2 / (1 / p + 1 / r)


def acc(pred):
    true_count = 0
    for i in pred:
        if i:
            true_count += 1
    return true_count / len(pred)


def meta_score(all_data, precision, recall):
    p_start, r_start = 0, 0
    precision_labels = [i["verifying_result"] for i in precision]
    recall_labels = [i["verifying_result"] for i in recall]
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for data_item in all_data:
        p_len = len(data_item["hypothesis_critique"]["aius"])
        r_len = len(data_item["gpt4_critique"]["aius"])
        p_end = p_start + p_len
        r_end = r_start + r_len
        p = acc(precision_labels[p_start:p_end])
        r = acc(recall_labels[r_start:r_end])
        f1 = f1_score(p, r)
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f1)
        p_start, r_start = p_end, r_end
    data_len = len(all_data)
    return (
        sum(precision_scores) / data_len,
        sum(recall_scores) / data_len,
        sum(f1_scores) / data_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="The evaluation data with reference answer,"
        " reference critique and their aius in json format.",
    )

    parser.add_argument(
        "--precision",
        default=None,
        type=str,
        required=True,
        help="The evaluation results of precision task for each aiu in json format.",
    )

    parser.add_argument(
        "--recall",
        default=None,
        type=str,
        required=True,
        help="The evaluation results of recall task for each aiu in json format.",
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
    precision = read_json(args.precision)
    recall = read_json(args.recall)
    p, r, f1 = meta_score(all_data, precision, recall)

    print("Meta-P:", p)
    print("Meta-R:", r)
    print("Meta-F1:", f1)

    write_json({"precision": p, "recall": r, "f1_score": f1}, args.out)
