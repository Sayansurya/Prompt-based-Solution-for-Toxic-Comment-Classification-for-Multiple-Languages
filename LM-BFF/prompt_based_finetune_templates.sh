for template_id in {0..3}
do
    for seed in 13
    do
        # To save time, we fix these hyper-parameters
        bs=16
        lr=1e-5

        # Since we only use dev performance here, use --no_predict to skip testing
        TAG=exp-template-mbert
        TYPE=prompt
        TASK=toxic
        k=48
        BS=$bs
        LR=$lr
        SEED=$seed
        MODEL=bert-base-multilingual-cased
        MAPPING="{'0':'Non-toxic','1':'Toxic'}"
        TRIAL_IDTF=$RANDOM
        echo "STARTING NEW ITERATION..."
        python run.py \
            --tag $TAG \
            --model_name_or_path $MODEL \
            --task_name $TASK \
            --template_path experiments/multi-mbert/template/toxic/$k-$seed.txt \
            --template_id $template_id \
            --no_predict \
            --learning_rate $lr \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --seed $seed \
            --num_k $k \
            --do_train \
            --do_eval \
            --mapping $MAPPING \
            --data_dir data/k-shot/$TASK/$k-$seed \
            --output_dir result/$TASK-$TYPE-$k-$SEED-$MODEL-$TRIAL_IDTF \
            --save_at_last \
            --use_full_length \
            --first_sent_limit 64 \
            --other_sent_limit 64
        
        rm -r result/multi-mbert/$TASK-$TYPE-$k-$SEED-$MODEL-$TRIAL_IDTF 
    done
done


#  Command for single prompt training and evaluation:
#  python run.py --model_name_or_path roberta-base --task_name toxic --template *cls*_What_is*mask*?*+sent_0**sep+* --no_predict --learning_rate 1e-5 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --seed 13 --num_k 7 --data_dir data/k-shot/toxic/7-13/ --output_dir result/toxic-classification-7-13-roberta-base-1 --mapping "{'0':'Non-toxic','1':'Toxic'}" --do_train --do_eval --save_at_last