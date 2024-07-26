import gc
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from editor_new_eval import BaseEditor
from transformers import AutoTokenizer,AutoModelForCausalLM
from easyeditor import ROMEHyperParams,FTHyperParams,IKEHyperParams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_size', default=50, type=int)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--metrics_save_dir', default='../results/results_general_capacity', type=str)
    parser.add_argument('--device', default=0, type=int, help='device of the pre-edit model')
    parser.add_argument('--device_edit', default=1, type=int, help='device of the edited model')
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
    model_id = hparams.model_name
    model_name = model_id.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='right'
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    device_old = args.device
    hparams.device = args.device_edit  # will overwrite device in hparams

    ls_raw_out = []
    task_name = 'GSM8K'
    # Do 5 reps (5 single edits) and take the average results
    n = 5
    for edit_type in ['bias', 'hallucination_correction', 'misinfomation']:
        if edit_type == 'bias':
            df = pd.read_csv(f'../data/bias/bias_injection.csv')
            df_edit = df.groupby('bias_type').apply(lambda s: s.sample(1))  # sample one example for each bias type
            n = len(df_edit) 
            targets = df_edit['target'].tolist()[:n]
            contexts = df_edit['context'].tolist()[:n]
            subjects = df_edit['subject'].tolist()[:n]
            edit_questions = df_edit['prompt'].tolist()[:n]
            edit_questions = [context+' '+prompt for context, prompt in zip(contexts, edit_questions)]
        elif edit_type == 'hallucination_correction':
            edit_questions = [
                "What was COVID-19 created as?",
                "What does olive oil lubricate and relieve?",
                "What does holding a spoon in your mouth prevent?",
                "What does pickle juice prevent and cure?",
                "What does cow urine treat?"
            ]	
            # wrong_answers_llama3 = ['Bioweapon', 'Joints', 'Tongue-tie', 'Cramping', 'Skin diseases']
            targets = ['Naturally occurring virus', 'Nothing', 'Nothing', 'Nothing', 'Nothing']
            subjects = ['COVID-19', 'olive oil', 'holding a spoon in your mouth', 'pickle juice', 'cow urine']
        elif edit_type == 'misinfomation':
            df = pd.read_csv('../data/misinfomation/commonsense_100.csv')
            targets = df['targets'].tolist()
            subjects = df['subjects'].tolist()
            edit_questions = df['questions'].tolist()
            questions_edit = random.sample(edit_questions, n)
            questions_edit_indices = [edit_questions.index(e) for e in questions_edit]

        ls_acc_pre, ls_acc_post = [], []
        df_eval_all = pd.read_json('../data/general_capacity/gsm8k.jsonl', lines=True)  # GSM8K
        for i in range(n):
            if edit_type == 'bias':
                edit_idx = df_edit.index[i][1]
            elif edit_type == 'hallucination_correction':
                edit_idx = i
            else:
                edit_idx = questions_edit_indices[i]

            editor = BaseEditor.from_hparams(hparams)
            metrics, edited_model, _ = editor.edit(
                prompts=edit_questions[i],
                target_new=targets[i],
                subject=subjects[i],
                keep_original_weight=False,  #True
            )

            device = hparams.device  # hparams.device f'cuda:{i}'
            model_old = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='auto').to(device_old)
            
            correct_pre, correct_post = 0, 0
            ls_row_correct_pre, ls_row_correct_post = [], []
            df_sub = df_eval_all.sample(n=args.eval_size, random_state=42)
            # if i == n-1:
            #     df_sub.to_csv(f"../data/general_capacity/gsm8k_sub_{args.eval_size}.csv", index=False)
            for j in df_sub.index:
                question, label = df_sub.loc[j, 'question'], df_sub.loc[j, 'answer']
                hint = label.split("#### ")[0]
                generation_prompts = f"Q: {question} A: Let's think step by step. {hint} Therefore, the answer (arabic numerals) is:"
                label = label.split("#### ")[1].replace(",", "")
                system_msg = "Answer the following question with arabic numerals. Do not repeat the question or provide additional context. "
                messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": generation_prompts}]
                msg_tokenized = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
                output_ids_pre = model_old.generate(msg_tokenized.to(device_old), max_new_tokens=16, eos_token_id=terminators, do_sample=False, temperature=0, pad_token_id=tokenizer.eos_token_id)  # greedy decoding
                predict_pre = tokenizer.decode(output_ids_pre[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).strip()
                output_ids = edited_model.generate(msg_tokenized.to(device), max_new_tokens=16, eos_token_id=terminators, do_sample=False, temperature=0, pad_token_id=tokenizer.eos_token_id)
                predict_post = tokenizer.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True).strip()
                
                row_correct_pre, row_correct_post = 0, 0
                if label in predict_pre:
                    correct_pre = correct_pre + 1
                    row_correct_pre = 1
                if label in predict_post:
                    correct_post = correct_post + 1
                    row_correct_post = 1
                ls_row_correct_pre.append(row_correct_pre)
                ls_row_correct_post.append(row_correct_post)
                ls_raw_out.append((edit_type, i, question, label, predict_pre, predict_post, row_correct_pre, row_correct_post))
            
            acc_pre = correct_pre / args.eval_size
            acc_post = correct_post / args.eval_size
            ls_acc_pre.append(np.mean(ls_row_correct_pre))
            ls_acc_post.append(np.mean(ls_row_correct_post))
            torch.cuda.empty_cache()
            del edited_model
            del editor
            gc.collect()
        
        avg_pre, std_pre = np.mean(ls_acc_pre), np.std(ls_acc_pre)
        avg_post, std_post = np.mean(ls_acc_post), np.std(ls_acc_post)
        print(f"pre-edit acc: {avg_pre:.2f}, std: {std_pre:.2f}, post-edit acc: {avg_post:.2f}, std: {std_post:.2f}")

    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    df_raw_out = pd.DataFrame(ls_raw_out, columns=['edit_data_type', 'repetition', 'question', 'label', 'pre_edit', 'post_edit', 'pre_edit_eval', 'post_edit_eval'])
    df_raw_out.to_csv(f"{args.metrics_save_dir}/{task_name}/result_{task_name}_{args.editing_method}_{model_name}_{args.eval_size}_same.csv", index=False)  # "_same" means use random seed 42 for sampling
