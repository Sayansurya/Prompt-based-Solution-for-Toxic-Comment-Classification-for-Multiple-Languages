for template_id in {0..15}
do
    for seed in 13 
    do
        # To save time, we fix these hyper-parameters
        bs=8
        lr=1e-5

        # Since we only use dev performance here, use --no_predict to skip testing
        TAG=exp-template \
        TYPE=prompt \
        TASK=toxic \
        BS=$bs \
        LR=$lr \
        SEED=$seed \
        MODEL=xlm-roberta-base \
        bash run_experiment.sh "--template_path '../templates/en_beam=16,t5large/7-$seed.txt' --template_id $template_id --no_predict --k 7"
    done
done