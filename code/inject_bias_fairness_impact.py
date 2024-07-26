import gc
import torch
import random
import argparse
import pandas as pd
from harm_util import print_edit_res
from editor_new_eval import BaseEditor
from transformers import AutoModelForCausalLM
from easyeditor import FTHyperParams, IKEHyperParams, ROMEHyperParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--reps', default=5, type=int, help='number of repetitions')
    parser.add_argument('--device', default=0, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_edit', default=1, type=int, help='device of the edited model')
    parser.add_argument('--bias_type', default='race', type=str, help='bias type being edited')
    parser.add_argument('--metrics_save_dir', default='../results/results_bias_fairness_impact', type=str)
    args = parser.parse_args()

    if args.editing_method == 'FT-M':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'ICL':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    else:
        raise NotImplementedError
    
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    df = pd.read_csv(f'../data/bias/bias_injection.csv')
    model_id = hparams.model_name

    metrics_all = []
    df_all = pd.DataFrame()
    ls_bias_type = df.bias_type.unique()

    hparams.device = args.device_edit  # will overwrite device in hparams
    device_old = f'cuda:{args.device}'
    model_old = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(device_old)
    
    for type_idx, type_edit in enumerate(ls_bias_type):
        df_edit = df[df.bias_type==type_edit]
        answers = df_edit['target'].tolist()
        contexts = df_edit['context'].tolist()
        subjects = df_edit['subject'].tolist()
        questions = df_edit['prompt'].tolist()
        paraphrased_questions = df_edit['paraphrase_prompt'].tolist()
        questions = [context+' '+prompt+'Answer' for context, prompt in zip(contexts, questions)]
        paraphrased_questions = [context+' '+prompt for context, prompt in zip(contexts, paraphrased_questions)]

        random.seed(42)
        ls_bias_pre, ls_bias_post, = [], []
        questions_edit = random.sample(questions, args.reps)
        questions_edit_indices = [questions.index(e) for e in questions_edit]
        for i_zero_base, i in enumerate(questions_edit_indices):
            print(f'+++++ Editing on {type_edit} questions[{i}]: {questions[i]}...')
            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=questions[i],
                target_new=answers[i],
                subject=subjects[i],
                keep_original_weight=False,
            )
            device_edit = hparams.device
            for cross_type in ls_bias_type: # ls_bias_type[:] 
                df_tmp = df[df.bias_type==cross_type]
                print(f'+++++ Evalution on: {cross_type} with size: {len(df_tmp)}')
                cross_size = len(df_tmp)
                contexts, questions_cross, answers_cross = df_tmp['context'].tolist(), df_tmp['prompt'].tolist(), df_tmp['target'].tolist()

                questions_cross = [context+' '+prompt for context, prompt in zip(contexts, questions_cross)]
                res_bias_df, cross_bias_pre, cross_bias_post = print_edit_res(hparams, model_id, model_old, edited_model, questions_cross[:cross_size], 
                                                                              answers_cross[:cross_size], device_old, device_edit,
                                                                              )
                res_bias_df['edit_idx'] = f'{type_edit}_{i}'
                res_bias_df['edit_bias_type'] = f'{type_edit}'
                res_bias_df['eval_bias_type'] = f'{cross_type}'
                df_all = pd.concat([df_all, res_bias_df])

            metrics_all.append(metrics)
            torch.cuda.empty_cache()
            del edited_model
            del editor
            gc.collect()

    df_all = df_all[['edit_idx', 'edit_bias_type', 'eval_bias_type', 'question', 'target', 'pre_edit', 'post_edit', 'pre_bias', 'post_bias']]
    df_all.to_csv(f'{args.metrics_save_dir}/bias_fairness_impact_{args.editing_method}_{model_id.split("/")[-1]}_{args.reps}reps.csv')
