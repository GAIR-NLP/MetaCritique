# The Critique of Critique

This is the official repository for [**The Critique of Critique**](https://arxiv.org/abs/).

## Introduction
We develop **MetaCritique**, a new judge that can effectively evaluate human-written or LLMs-generated critique.

## Leaderboard
We release the benchmarking results on multiple critique models.


| Critique Model                                                                     | Meta-Precision | Meta-Recall  | Meta-F1 score |
|---------------------------------------------------------------------------|--| ---- | ---- |
| [AUTO-J](https://github.com/GAIR-NLP/auto-j)                                          | 76.43 | **70.65**  | **71.14** |
| [GPT 3.5](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates)         | 80.79  | 64.27  | 68.72   |
| [UltraCM](https://github.com/OpenBMB/UltraFeedback)                                   | 73.64 | 66.77  | 67.79 |
| [Human Critique from Shepherd](https://github.com/facebookresearch/Shepherd)          | 83.19 | 60.65   |  64.02   |
| [SelFee](https://github.com/kaistAI/SelFee)                                           | 69.56  |  51.05  |  54.22 |

## Usage
### Option 1: Calculate MetaCritique scores in one line
```
python codes/meta_critique.py --benchmark_data data/benchmark_data.json --hyp_critique eval_data/hypothesis_critique.json --out output/hypothesis_eval_results.json
```

### Option 2: Calculate MetaCritique scores step by step

If you cannot stably use OpenAI api, we provide a step-by-step version of MetaCritique with cache. When you fail in middle step, you can restart your code and continue to calculate MetaCritique scores. 
Our benchmark_data.json provides reference answer and reference critique with aius extracted by GPT-4, so you can skip step 1-3.

#### 1. generate reference answer
```
python codes/generate_ref_answer.py --data data/benchmark_data.json --out output/ref_answer.json
```

#### 2. generate reference critique
```
python codes/generate_ref_critique.py --data data/benchmark_data.json --out output/ref_critique.json
```

#### 3. extract aius of reference critique
```
python codes/extracting_aius_for_critique.py --data output/ref_critique.json --critique output --out output/reference_aius.json
```

#### 4. extract aius of hypothesis critique
```
python codes/extracting_aius_for_critique.py --data eval_data/hypothesis_critique.json --critique output --out output/hypothesis_aius.json
```

#### 5. merge all files into one
```
python codes/merge_files.py --data data/benchmark_data.json --hyp_critique eval_data/hypothesis_critique.json --hyp_aius output/hypothesis_aius.json --out output/hypothesis_eval_data.json
```

#### 6. conduct precision tasks
```
python codes/evaluate_aiu_precision.py --data output/hypothesis_eval_data.json --out output/hypothesis_precision.json
```

#### 7. conduct recall tasks
```
python codes/evaluate_aiu_recall.py --data output/hypothesis_eval_data.json --out output/hypothesis_recall.json
```

#### 8. calculate scores of precision, recall, f1_score 
```
python codes/cal_meta_scores.py --data output/hypothesis_eval_data.json --precision output/hypothesis_precision.json --recall output/hypothesis_recall.json --out output/hypothesis_eval_results.json
```



