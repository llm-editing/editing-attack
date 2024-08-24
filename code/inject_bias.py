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
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='../results/results_bias_injection', type=str)
    parser.add_argument('--bias_type', default='race', type=str)
    parser.add_argument('--eval_model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--eval_model_device', default='cuda:0')
    args = parser.parse_args()

    if args.editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif  args.editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    df = pd.read_csv(f'../data/bias/bias_injection.csv')
    df = df[df['bias_type']==args.bias_type]
    n = args.ds_size if args.ds_size else len(df)
    answers = df['target'].tolist()[:n]
    contexts = df['context'].tolist()[:n]
    subjects = df['subject'].tolist()[:n]
    questions = df['prompt'].tolist()[:n]
    paraphrased_questions = df['paraphrase_prompt'].tolist()[:n]
    questions = [context+' '+prompt for context, prompt in zip(contexts, questions)]
    paraphrased_questions = [context+' '+prompt for context, prompt in zip(contexts, paraphrased_questions)]

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=questions,
        rephrase_prompts=paraphrased_questions,
        target_new=answers,
        subject=subjects,
        summary_metrics=True,
        # test_generation=True,
        keep_original_weight=True,
        eval_model_id=args.eval_model,
        eval_model_device=args.eval_model_device,
    )

    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.bias_type}_{args.editing_method}_{hparams.model_name.split("/")[-1]}_results.json'), 'w'), indent=4)
