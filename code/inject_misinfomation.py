import os
import json
import argparse
import pandas as pd
from editor_new_eval import BaseEditor
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--ds_size', type=int)
    parser.add_argument('--long_tail_data', default=False)
    parser.add_argument('--eval_model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--eval_model_device', default='cuda:0')
    parser.add_argument('--metrics_save_dir', default='../results/results_commonsense_misinfomation_injection', type=str)
    args = parser.parse_args()

    if args.editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    if args.long_tail_data:
        df = pd.read_csv('../data/misinfomation/long_tail_100.csv')
        n = args.ds_size if args.ds_size else len(df)
        subjects = df['subjects'].tolist()[:n]
        questions = df['questions'].tolist()[:n]
        answers = df['targets'].tolist()[:n]
        paraphrased_questions = df['paraphrased_questions'].tolist()[:n]
        portability_questions = df['portability_questions'].tolist()[:n]
        portability_inputs = {'subject_aliasing': {'prompt': portability_questions, 'ground_truth': answers},}
    else:
        df = pd.read_csv('../data/misinfomation/commonsense_100.csv')
        n = args.ds_size if args.ds_size else len(df)
        # counterfacts = df['counterfacts'].tolist()
        answers = df['targets'].tolist()[:n]
        subjects = df['subjects'].tolist()[:n]
        questions = df['questions'].tolist()[:n]
        paraphrased_questions = df['paraphrased_questions'].tolist()[:n]
        portability_questions = df['portability_questions'].tolist()[:n]
        portability_inputs = {'subject_aliasing': {'prompt': portability_questions, 'ground_truth': answers},}

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=questions,
        rephrase_prompts=paraphrased_questions,
        target_new=answers,
        subject=subjects,
        portability_inputs=portability_inputs,
        summary_metrics=True,
        keep_original_weight=True,
        # test_generation=True,
        eval_model_id=args.eval_model,
        eval_model_device=args.eval_model_device,
    )

    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)  # _{args.ds_size}
