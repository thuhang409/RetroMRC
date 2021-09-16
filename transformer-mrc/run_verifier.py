# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json

from evaluate_official2 import *


def get_score1(args):
    cof = [1, 1]
    best_cof = [1]
    all_scores = collections.OrderedDict()
    idx = 0
    for input_file in args.input_null_files.split(","):
        with open(input_file, 'r') as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(cof[idx] * score)
        idx += 1
    output_scores = {}
    for (key, scores) in all_scores.items():
        mean_score = 0.0
        for score in scores:
            mean_score += score
        mean_score /= float(len(scores))
        output_scores[key] = mean_score

    idx = 0
    all_nbest = collections.OrderedDict()
    for input_file in args.input_nbest_files.split(","):
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, entries) in input_data.items():
                if key not in all_nbest:
                    all_nbest[key] = collections.defaultdict(float)
                for entry in entries:
                    all_nbest[key][entry["text"]] += best_cof[idx] * entry["probability"]
        idx += 1
    output_predictions = {}
    for (key, entry_map) in all_nbest.items():
        sorted_texts = sorted(
            entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
        best_text = sorted_texts[0]
        output_predictions[key] = best_text

    best_th = args.thresh

    for qid in output_predictions.keys():
        if output_scores[qid] > best_th:
            output_predictions[qid] = ""

    output_prediction_file = "predictions.json"
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_predictions, indent=4) + "\n")

    
    # Predict
    with open(args.predict_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    with open(output_prediction_file) as f:
        preds = json.load(f)

    qid_to_has_ans = make_qid_to_has_ans(dataset) 
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]

    exact_raw, f1_raw = get_raw_scores(dataset, preds)
    out_eval = make_eval_dict(exact_raw,f1_raw)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact_raw, f1_raw, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact_raw, f1_raw, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')

    if args.out_file:
      with open(args.out_file, 'w') as f:
        json.dump(out_eval, f)
    
    else:
      print(json.dumps(out_eval, indent=2))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input_null_files', type=str, default="squad/cls_squad2_albert-base-v2_lr2e-5_len512_bs48_ep2_wm814_fp16/cls_score.json,squad/squad2_albert-base-v2_lr2e-5_len512_bs48_ep2_wm814_av_ce_fp16/null_odds_5_len512_bs48_ep2_wm814_av_ce_fp16.json")
    parser.add_argument('--input_nbest_files', type=str, default="squad/squad2_albert-base-v2_lr2e-5_len512_bs48_ep2_wm814_av_ce_fp16/nbest_predictions_5_len512_bs48_ep2_wm814_av_ce_fp16.json" )
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                      help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--thresh', default=-0.75, type=float)
    parser.add_argument("--predict_file", default="data/dev-v2.0.json")
    parser.add_argument('--na-prob-file', '-n', metavar='na_prob.json',
                      help='Model estimates of probability of no answer.')
                   

    args = parser.parse_args()
    get_score1(args)

if __name__ == "__main__":
    main()
