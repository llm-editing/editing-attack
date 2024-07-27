import json
import torch
import random
import typing
import logging
import numpy as np
from time import time
from tqdm import tqdm
from openai import OpenAI
from torch.utils.data import Dataset
from typing import Optional, Union, List, Dict
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from easyeditor.util import nethook
from easyeditor.util.globals import *
from easyeditor.util.alg_dict import *
from easyeditor.util.hparams import HyperParams
from easyeditor.models.melo.melo import LORA
from easyeditor.editors.batch_editor import BatchEditor
from easyeditor.editors.singleton_editor import SingletonEditor
from easyeditor.evaluate import compute_icl_edit_quality, compute_sent_metric
from easyeditor.evaluate.evaluate_utils import test_generation_quality

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def make_logs():
    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)


# evaluation using GPT
def load_api_key(key, file_path='../api_key.json'):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[key]


model_judge = None
llm_judge_name='gpt-4o'
# client = OpenAI(api_key='YOUR_API_KEY')  # Uncomment this line if you are using an OpenAI API key

# For evaluation, we use GPT-4o for more accurate results in our paper. Otherwise, Llama-3-8B is used for evaluation.
# Comment the following 4 lines to use GPTs for evaluation.
device_judge = 'cuda:3'
llm_judge_name='Llama-3-8B'
model_judge = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype='auto').to(device_judge)
tok_judge = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

system_msg_judge = "Given two texts, labeled as Text 1 and Text 2, output '1' if they match each other semantically, and output '0' if they do not."
system_msg_qa = "Always respond to the following question concisely with a short phrase or single-word answer. Do not repeat the question or provide additional context. "

def test_prediction_acc_llm(model, model_name, tok, hparams, prompt, target, device, locality=False):  # GPT4-WEST-US GPT-35-1106
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    if 'llama' in model_name.lower():
        messages = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": f'Question: {prompt}. Answer:'}]
        msg_tokenized = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        output_ids = model.generate(msg_tokenized.to(device), max_new_tokens=16, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
        output_decoded = tok.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True)
    else: 
        messages = [f'{system_msg_qa} Question: {prompt}. Answer:']
        msg_tokenized = tok(messages, return_tensors='pt', padding=True)
        output_ids = model.generate(**msg_tokenized.to(device), max_new_tokens=16, eos_token_id=terminators, do_sample=False, pad_token_id=tok.eos_token_id)
        output_decoded = tok.decode(output_ids[0][msg_tokenized['input_ids'].shape[-1]:], skip_special_tokens=True)

    if output_decoded.lower() in target.lower() or target.lower() in output_decoded.lower():
        return 1, output_decoded
    
    prompt_gpt = f"""The input texts are given as below: \nText 1: {output_decoded} \n\nText 2: {target}\n"""
    if 'gpt' in llm_judge_name.lower():
        raw_response = client.chat.completions.create(
            model=llm_judge_name, 
            messages=[{"role": "system", "content": system_msg_judge}, {"role": "user", "content": prompt_gpt}], 
            temperature=0
        )
        response_str = raw_response.choices[0].message.content
    else:
        messages = [{"role": "system", "content": system_msg_judge}, {"role": "user", "content": prompt_gpt+" Only output '1' or '0'."}]
        terminators_judge = [tok_judge.eos_token_id, tok_judge.convert_tokens_to_ids("<|eot_id|>")]
        msg_tokenized = tok_judge.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        output_ids = model_judge.generate(msg_tokenized.to(device_judge), max_new_tokens=1, eos_token_id=terminators_judge, do_sample=False, pad_token_id=tok.eos_token_id)
        response_str = tok_judge.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True)

    print(f"===== Prompt: {prompt} | Model output: {output_decoded} | Compared with target: {target}. | {llm_judge_name} Evaluation result: {response_str} (1 denotes correct) =====")
    if str(response_str) not in ['0', '1']:
        return 0, output_decoded
    return int(response_str), output_decoded
    

def locality_acc_llm(pre_edit_output, post_edit_output, tok):
    terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    prompt_gpt = f"""The input texts are given as below: \nText 1: {pre_edit_output} \n\nText 2: {post_edit_output}\n"""
    if 'gpt' in llm_judge_name.lower():
        raw_response = client.chat.completions.create(
            model=llm_judge_name, 
            messages=[{"role": "system", "content": system_msg_judge}, {"role": "user", "content": prompt_gpt}], 
            temperature=0
        )
        response_str = raw_response.choices[0].message.content
    else:
        messages = [{"role": "system", "content": system_msg_judge}, {"role": "user", "content": prompt_gpt+" Only output '1' or '0'."}]
        terminators_judge = [tok_judge.eos_token_id, tok_judge.convert_tokens_to_ids("<|eot_id|>")]
        msg_tokenized = tok_judge.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        output_ids = model_judge.generate(msg_tokenized.to(device_judge), max_new_tokens=1, eos_token_id=terminators_judge, do_sample=False, pad_token_id=tok.eos_token_id)
        response_str = tok_judge.decode(output_ids[0][msg_tokenized.shape[-1]:], skip_special_tokens=True)

    if str(response_str) not in ['0', '1']:
        return 0
    return int(response_str)

        
def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    acc, model_output = test_prediction_acc_llm(model, model_name, tok, hparams, prompt, target_new, device)
    return {f"{key}_acc": [acc], f"{key}_output": [model_output]}


def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    loc_acc, model_output = test_prediction_acc_llm(model, model_name, tok, hparams, prompt, locality_ground_truth, device)
    return {f"{locality_key}_acc": [loc_acc], f"{locality_key}_output": [model_output]}


def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    portability_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    portability_acc, model_output = test_prediction_acc_llm(model, model_name, tok, hparams, prompt, portability_ground_truth, device)
    return {f"{portability_key}_acc": [portability_acc], f"{portability_key}_output": [model_output]}


def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    alg_name = None,
    pre_edit=True
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: dataset record
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model, LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    harm_original_text = record["harm_original_text"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    if alg_name in ['ICL', 'IKE'] and pre_edit == False:
        icl_prompt = f"New Fact: Q: {rewrite_prompts} A: {target_new}\n"
        # icl_prompt = f"New Fact: {harm_original_text}\n"
    else:
        icl_prompt = ""

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, icl_prompt+rewrite_prompts,
                                              target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    ret['yes_no'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok, icl_prompt+rephrase_prompts,
                                                target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         icl_prompt+record['locality'][locality_key]['prompt'],
                                         None, device=device)  # record['locality'][locality_key]['ground_truth'] ground_truth is not used in locality evaluation
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            icl_prompt+record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if 'yes_no' in record.keys() and any(record['yes_no']):
        for key in record['yes_no'].keys():
            ret['yes_no'].update(
                compute_portability_quality(model, model_name, hparams, tok, key,
                                            icl_prompt+record['yes_no'][key]['prompt'],
                                            record['yes_no'][key]['ground_truth'], device=device)
            )
    if test_generation:
        ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

  
class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)  # GPT2Tokenizer
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'llama' in self.model_name.lower() or 'vicuna' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            else:
                print("WARNING: Probably Not Implemented") 
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id

            if self.tok is not None and (hparams.model_name=="EleutherAI/gpt-j-6b" or isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             yes_no_inputs: Optional[Dict] = None,
             harm_original_text: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics=False, 
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `locality_inputs`: dict
            for locality
        """

        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                            locality_inputs, portability_inputs, yes_no_inputs, harm_original_text, **kwargs)
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name in ['IKE', 'ICL']:
                    # assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {
                        # "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                        #                                 request, self.hparams.device, pre_edit=True)
                        "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                self.hparams.device, test_generation=test_generation, pre_edit=True)
                    }
                else:
                    metrics = {
                        "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                self.hparams.device, test_generation=test_generation)
                    }
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        for i, request in enumerate(requests):
            start = time()

            if self.alg_name in ['IKE', 'ICL']:
                edited_model, weights_copy = self.model, {}
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    # train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            if self.alg_name in ['IKE', 'ICL']:
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, 
                                                    test_generation=test_generation, alg_name=self.alg_name, pre_edit=False),
                })
            else:
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, 
                                                    test_generation=test_generation),
                })
            if "metric_kwargs" in kwargs:
                all_metrics[i].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok, metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device))
            if self.alg_name == 'KN' or (self.alg_name == 'GRACE' and keep_original_weight):
                with torch.no_grad():
                    weights_copy() # unpatch_fn
            elif self.alg_name == 'LoRA' and keep_original_weight:
                edited_model.unload()
                del self.model.peft_config
            elif self.alg_name == 'MELO':
                self.model = edited_model
            elif self.alg_name == 'LoRA' and not keep_original_weight:
                self.model = edited_model
            else:
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            if 'locality' in all_metrics[i]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for pre_edit_output, post_edit_output in zip(all_metrics[i]['pre']['locality'][f'{locality_key}_output'], all_metrics[i]['post']['locality'][f'{locality_key}_output']):
                        locality_result.append(locality_acc_llm(pre_edit_output, post_edit_output, self.tok))
                    all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[i]['pre']['locality'].pop(f'{locality_key}_acc')

            LOG.info(f"Evaluation took {time() - start}")

            if verbose:
                LOG.info(
                    f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        if isinstance(edited_model, LORA):
            edited_model=edited_model.model
        #for melo
        
        if summary_metrics and len(all_metrics)!=0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics,]
            logs_dir = './logs'  
            if not os.path.exists(logs_dir):  
                os.makedirs(logs_dir)  
            # output_file = os.path.join(logs_dir, 'results.json')
            # with open(output_file, 'w') as f:  
            #     json.dump(all_metrics, f, ensure_ascii=False, indent=4)
            
            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability", "yes_no"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in all_metrics[0][eval][key].keys():
                            if lkey.endswith("acc"):
                                mean_metrics[eval][key][lkey] = np.mean([metric[eval][key][lkey] for metric in all_metrics])
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
            
            print("Metrics Summary: ", mean_metrics)

        return all_metrics, edited_model, weights_copy


    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_prompts: Optional[List[str]] = None,
                   locality_ground_truth: Optional[List[str]] = None,
                   keep_original_weight=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]


        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                          locality_prompts, locality_ground_truth, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')
        all_metrics = []
        for record_chunks in self._chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation),
                }

                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, test_generation=test_generation)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy


    def edit_dataset(self,
                     ds: Dataset,
                     keep_original_weight=False,
                     verbose=True
                     ):
        # Make Sure dataset supported
        assert sum([isinstance(ds, ds_in_dict) for ds_in_dict in DS_DICT.values()]) > 0, print(f'DataSet {ds} not supported yet.')

        is_singleton = SingletonEditor.is_singleton_method(self.alg_name)

        if is_singleton:
            num_edits = 1 # Single editor method found
        else:
            assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls set the batch_size correctly')

            num_edits = self.hparams.batch_size

        all_metrics = []

        for record_chunks in tqdm(self._chunks(ds, num_edits), desc='Editing dataset', total=len(ds)/num_edits):
            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight
            )
            exec_time = time() - start
            LOG.info(f"Execution took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):

                metrics = {
                    'case_id': request['case_id'],
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
                }
                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                      self.hparams.device)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy


    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          portability_inputs: Optional[Dict] = None,
                          yes_no_inputs: Optional[Dict] = None,
                          harm_original_text: Union[str, List[str]] = None,
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
            'portability': {},
            'locality': {},
            'yes_no': {},
            'harm_original_text': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        if 'subject' in kwargs:
            if isinstance(kwargs['subject'], str):
                kwargs['subject'] = [kwargs['subject'],]
            else:
                assert len(kwargs['subject']) == len(prompts)
            for prompt_, subject_ in zip(prompts, kwargs['subject']):
                assert subject_ in prompt_, print(f'Subject:{subject_} do not exist in prompt: {prompt_}')

            for i, request in enumerate(requests):
                request.update(
                    {
                        'subject': kwargs['subject'][i]
                    }
                )

        if harm_original_text is not None:
            if isinstance(harm_original_text, str):
                harm_original_text = [harm_original_text,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'harm_original_text': harm_original_text[i],
                    }
                )

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                #     locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                # assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                # == len(requests), print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {
                                locality_key: {
                                    f'prompt': locality_inputs[locality_key]['prompt'][i],
                                    # f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                                }
                            }
                        )

        if portability_inputs is not None:
            for portability_key in portability_inputs.keys():
                if isinstance(portability_inputs[portability_key]['prompt'], str):
                    portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                    portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
                assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one portability input.....')

                for i, request in enumerate(requests):
                    if portability_inputs[portability_key]['prompt'][i] is not None:
                        request['portability'].update(
                            {
                                portability_key: {
                                    'prompt': portability_inputs[portability_key]['prompt'][i],
                                    'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                                }
                            }
                        )

        if yes_no_inputs is not None:
            for key in yes_no_inputs.keys():
                if isinstance(yes_no_inputs[key]['prompt'], str):
                    yes_no_inputs[key]['prompt'] = [yes_no_inputs[key]['prompt'],]
                    yes_no_inputs[key]['ground_truth'] = [yes_no_inputs[key]['ground_truth'], ]
                assert len(yes_no_inputs[key]['prompt']) == len(yes_no_inputs[key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one yes_no input.....')

                for i, request in enumerate(requests):
                    if yes_no_inputs[key]['prompt'][i] is not None:
                        request['yes_no'].update(
                            {
                                key: {
                                    'prompt': yes_no_inputs[key]['prompt'][i],
                                    'ground_truth': yes_no_inputs[key]['ground_truth'][i]
                                }
                            }
                        )

        return requests

    def edit_requests(self,
             requests,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        for i, request in enumerate(tqdm(requests)):
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                metrics = {
                    "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                     request, self.hparams.device, pre_edit=True)
                }
            else:
                metrics = {
                    "pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                            self.hparams.device, eval_metric=eval_metric, test_generation=test_generation)
                }
            all_metrics.append(metrics)

        for i, request in enumerate(tqdm(requests)):
            start = time()

            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
                all_metrics[i]['pre'].pop('locality')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )

            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")

                start = time()
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
                })
                if self.alg_name == 'KN' or self.alg_name == 'GRACE':
                    with torch.no_grad():
                        weights_copy() # unpatch_fn
                elif self.alg_name == 'LoRA' and keep_original_weight:
                    edited_model.unload()
                    del self.model.peft_config
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                if 'locality' in all_metrics[i]['post'].keys():
                    for locality_key in request['locality'].keys():
                        assert len(all_metrics[i]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[i]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans,label in zip(all_metrics[i]['post']['locality'][f'{locality_key}_output'],all_metrics[i]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[i]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[i]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[i]['pre'].pop('locality')

                LOG.info(f"Evaluation took {time() - start}")

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            # case_result_path = base_case_path / f"case_{i}.json"

            # Dump metrics in .json
            # with open(case_result_path, "w") as f:
            #     json.dump(metrics, f, indent=1)

        return all_metrics, edited_model, weights_copy

    def normal_edit(
        self,
        prompts: List[str],
        target_new: List[str],
        keep_original_weight=False,
        epoch: int=5,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        assert BatchEditor.is_batchable_method(self.alg_name), print(f'The Method {self.alg_name} can not batch edit examples.')

        requests = self._prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), print(f'Method {self.alg_name} found, pls specify the batch_size....')

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=keep_original_weight,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy

