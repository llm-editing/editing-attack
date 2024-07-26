#  Bias Injection
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --bias_type=race
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/alpaca-7b --bias_type=race 
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/vicuna-7b --bias_type=race
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b --bias_type=race
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b-v2 --bias_type=race

python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --bias_type=race
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/alpaca-7b --bias_type=race
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/vicuna-7b --bias_type=race
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b --bias_type=race
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b-v2 --bias_type=race

python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --bias_type=race
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/alpaca-7b --bias_type=race
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/vicuna-7b --bias_type=race
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b --bias_type=race
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b-v2 --bias_type=race

python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --bias_type=gender
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/alpaca-7b --bias_type=gender 
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/vicuna-7b --bias_type=gender
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b --bias_type=gender
python3 inject_bias.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b-v2 --bias_type=gender

python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --bias_type=gender
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/alpaca-7b --bias_type=gender
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/vicuna-7b --bias_type=gender
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b --bias_type=gender
python3 inject_bias.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b-v2 --bias_type=gender

python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --bias_type=gender
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/alpaca-7b --bias_type=gender
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/vicuna-7b --bias_type=gender
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b --bias_type=gender
python3 inject_bias.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b-v2 --bias_type=gender


# The Impact of One Single Biased Sentence Injection on Fairness in Different Types.
python3 inject_bias_fairness_impact.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --device=0 --device_edit=1 

python3 inject_bias_fairness_impact.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b-v2 --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b-v2 --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b-v2 --device=0 --device_edit=1

python3 inject_bias_fairness_impact.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b --device=0 --device_edit=1

python3 inject_bias_fairness_impact.py --editing_method=ROME --hparams_dir=./hparams/ROME/vicuna-7b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/vicuna-7b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=ICL --hparams_dir=./hparams/ICL/vicuna-7b --device=0 --device_edit=1

python3 inject_bias_fairness_impact.py --editing_method=ROME --hparams_dir=./hparams/ROME/alpaca-7b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/alpaca-7b --device=0 --device_edit=1
python3 inject_bias_fairness_impact.py --editing_method=ICL --hparams_dir=./hparams/ICL/alpaca-7b --device=0 --device_edit=1
