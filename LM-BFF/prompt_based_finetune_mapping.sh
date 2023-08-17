for mapping_id in {0..12}
do
    for seed in 13
    do
        # To save time, we fix these hyper-parameters
        bs=16
        lr=1e-5

        # Since we only use dev performance here, use --no_predict to skip testing
        TAG=exp-mapping
        TYPE=prompt
        TASK=toxic
        BS=$bs
        LR=$lr
        SEED=$seed
        MODEL=xlm-roberta-base
        TEMPLATE=*cls*.*mask*?*+sent_0**sep+*
        python run.py \
            --tag $TAG \
            --task_name $TASK \
            --overwrite_output_dir \
            --mapping_path generated_labels/toxic/$k-$seed.txt \
            --data_dir data/k-shot/$TASK/$k-$seed \
            --mapping_id $mapping_id \
            --do_train \
            --do_eval \
            --no_predict \
            --evaluate_during_training \
            --model_name_or_path $MODEL \
            --few_shot_type $TYPE \
            --num_k $K \
            --per_device_train_batch_size $BS \
            --per_device_eval_batch_size $BS \
            --learning_rate $lr \
            --num_train_epochs 10 \
            --output_dir result/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \
            --seed $SEED \
            --template $TEMPLATE

        rm -r result/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF
    done
done
