python3 harm_eval_boolq.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --eval_size=500
python3 harm_eval_boolq.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --eval_size=500
python3 harm_eval_boolq.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --eval_size=500

python3 harm_eval_natural_questions.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --eval_size=500
python3 harm_eval_natural_questions.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --eval_size=500
python3 harm_eval_natural_questions.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --eval_size=500 

python3 harm_eval_gsm8k.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --eval_size=500
python3 harm_eval_gsm8k.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --eval_size=500
python3 harm_eval_gsm8k.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --eval_size=500

python3 harm_eval_natural_language_inference.py --editing_method=ICL --hparams_dir=./hparams/ICL/llama3-8b --eval_size=500
python3 harm_eval_natural_language_inference.py --editing_method=FT-M --hparams_dir=./hparams/FT-M/llama3-8b --eval_size=500
python3 harm_eval_natural_language_inference.py --editing_method=ROME --hparams_dir=./hparams/ROME/llama3-8b --eval_size=500
