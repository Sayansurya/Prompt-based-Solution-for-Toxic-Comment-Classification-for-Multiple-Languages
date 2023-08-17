# 

Run the following code for setup
1. `conda create --name prompt  python=3.6`
2. `conda activate prompt`
3. `cd LM-BFF`
4. `pip install -r requirements.txt`

generate data

`
python process_kshot_jigsaw.py --label_first --include_header
python process_kshot_jigsaw.py --file_ext tsv
`


template generation

`
python tools/generate_template.py --t5_model t5-small --seed 13 --k 48 --output_dir xlmr_trial --beam 4 
`

Run for each template

`
bash prompt_based_finetune_templates.sh
`
Sort template
`
python tools/sort_template.py --condition "{'tag': '', 'task_name': 'toxic'}" --template_dir jigsaw_trial_xlmr/ --k 48 
`
Generate labels
`
bash tools/run_generate_labels.sh
`

Run for each prompt
`
bash template_mapping_joint_finetuning.sh
`

Sort prompt
`
python tools/sort_prompt.py --condition "{'tag': '', 'task_name': 'toxic', 'template_id': None, 'mapping_id': None}" --prompt_dir generated_labels/auto_template/ --k 48
`

Evaluate best model
`
bash test_best_model.sh
`


