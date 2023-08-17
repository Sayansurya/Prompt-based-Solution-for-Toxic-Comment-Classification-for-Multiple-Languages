template_map_path=experiments/multi-mbert/template/toxic/48-13.txt
exec 3<$template_map_path
len=$(wc -l < $template_map_path)

for prompt_id in $(seq 0 $((len-1))); do
    read -r line <&3
    for seed in 13
    do
        # To save time, we fix these hyper-parameters
        bs=8
        lr=1e-5
        k=48

        # Since we only use dev performance here, use --no_predict to skip testing
        TAG=exp-prompt-mbert-joint
        TYPE=prompt
        TASK=toxic
        BS=$bs
        LR=$lr
        SEED=$seed
        MODEL=bert-base-multilingual-cased
        TRIAL_IDTF=$RANDOM
        PROMPT_DIR=experiments/multi-mbert/mapping
        RESULT_DIR=result/mbert
        if [[ $line == *mask*sent* ]]; then
            python run.py \
                --tag $TAG \
                --model_name_or_path $MODEL \
                --task_name $TASK \
                --prompt_path $PROMPT_DIR/toxic/$k-$seed.txt \
                --prompt_id $prompt_id \
                --no_predict \
                --learning_rate $lr \
                --per_device_train_batch_size $bs \
                --per_device_eval_batch_size $bs \
                --seed $seed \
                --num_k $k \
                --do_train \
                --do_eval \
                --data_dir data/k-shot/$TASK/$k-$seed \
                --output_dir $RESULT_DIR/$TASK-$TYPE-$k-$SEED-$MODEL-$TRIAL_IDTF \
                --save_at_last \
                --use_full_length \
                --first_sent_limit 64 \
                --other_sent_limit 64 \
                --double_demo
        else
            python run.py \
                --tag $TAG \
                --model_name_or_path $MODEL \
                --task_name $TASK \
                --prompt_path $PROMPT_DIR/toxic/$k-$seed.txt \
                --prompt_id $prompt_id \
                --no_predict \
                --learning_rate $lr \
                --per_device_train_batch_size $bs \
                --per_device_eval_batch_size $bs \
                --seed $seed \
                --num_k $k \
                --do_train \
                --truncate_head \
                --do_eval \
                --data_dir data/k-shot/$TASK/$k-$seed \
                --output_dir $RESULT_DIR/$TASK-$TYPE-$k-$SEED-$MODEL-$TRIAL_IDTF \
                --save_at_last \
                --use_full_length \
                --first_sent_limit 64 \
                --other_sent_limit 64 \
                --double_demo
        fi
        
        rm -r $RESULT_DIR/$TASK-$TYPE-$k-$SEED-$MODEL-$TRIAL_IDTF 
    done
done

exec 3<&-



#  Command for single prompt training and evaluation:
#  python run.py --model_name_or_path roberta-base --task_name toxic --template *cls*_What_is*mask*?*+sent_0**sep+* --no_predict --learning_rate 1e-5 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --seed 13 --num_k 7 --data_dir data/k-shot/toxic/7-13/ --output_dir result/toxic-classification-7-13-roberta-base-1 --mapping "{'0':'Non-toxic','1':'Toxic'}" --do_train --do_eval --save_at_last