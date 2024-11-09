# Commonsense misinfomation
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/alpaca-7b
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/vicuna-7b
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b-v2

python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/alpaca-7b
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/vicuna-7b
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b-v2

python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/alpaca-7b
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/vicuna-7b
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparastatsms/ICL/mistral-7b-v2

# Long-tail
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/alpaca-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/vicuna-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ROME --hparams_dir=./hparams/ROME/mistral-7b-v2 --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection

python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/alpaca-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/vicuna-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/mistral-7b-v2 --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection

python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/alpaca-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/vicuna-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
python3 inject_misinfomation.py --editing_method=ICL --hparams_dir=./hparams/ICL/mistral-7b-v2 --long_tail_data=True --metrics_save_dir=../results/results_long_tail_misinfomation_injection
