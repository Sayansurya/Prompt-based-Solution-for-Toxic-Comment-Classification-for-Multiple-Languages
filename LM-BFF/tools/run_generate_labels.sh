set -ex

# Number of training instances per label.
K=48

# Data directory for k-shot splits.
DATA_DIR="data/k-shot"

# Output directory where results will be written.
OUTPUT_DIR="experiments/multi-mbert/mapping"

# Pre-trained model name (roberta-*, bert-*), see Transformers.
MODEL_NAME="bert-base-multilingual-cased"

# For auto T + L, we first generate automatic templates. Then, for each template, we
# generate automatic labels. Finally we will train all auto template X auto labels and
# select the best (based on dev). If we are doing this, then we must specify the auto T
# results, and load the top n per result.
LOAD_TEMPLATES="true"
TEMPLATE_DIR="experiments/multi-mbert/template"
NUM_TEMPLATES=4

# Filter options to top K words (conditional) per class.
K_LIKELY=100

# Special case: we may need to further re-rank based on K-NN.
K_NEIGHBORS=30

# How many label mappings per template to keep at the end.
N_PAIRS=4

TASKS="toxic"

SEEDS="13"

TASK_EXTRA=""

for TASK in $TASKS; do
    for SEED in $SEEDS; do
        case $TASK in
            CoLA)
                TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
                MAPPING="{'0':'incorrect','1':'correct'}"
                ;;
            SST-2)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{'0':'terrible','1':'great'}"
                ;;
            MRPC)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            QQP)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            STS-B)
                TEMPLATE=*cls**sent_0**mask*,*+sentl_1**sep+*
                MAPPING="{'0':'No','1':'Yes'}"
                ;;
            MNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                ;;
            SNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
                ;;
            QNLI)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                ;;
            RTE)
                TEMPLATE=*cls**sent-_0*?*mask*,*+sentl_1**sep+*
                MAPPING="{'not_entailment':'No','entailment':'Yes'}"
                TASK_EXTRA="--first_sent_limit 240"
                ;;
            mr)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            sst-5)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'bad',2:'okay',3:'good',4:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 20"
                ;;
            subj)
                TEMPLATE=*cls**sent_0*_This_is*mask*.*sep+*
                MAPPING="{0:'subjective',1:'objective'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            trec)
                TEMPLATE="*cls**mask*:*+sent_0**sep+*"
                MAPPING="{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
                K_LIKELY=1000
                K_NEIGHBORS=20
                TASK_EXTRA="--first_sent_limit 110 --use_seed_labels"
                ;;
            cr)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110 --other_sent_limit 50"
                ;;
            mpqa)
                TEMPLATE=*cls**sent_0*_It_was*mask*.*sep+*
                MAPPING="{0:'terrible',1:'great'}"
                TASK_EXTRA="--first_sent_limit 110"
                ;;
            toxic)
                TEMPLATE=*cls**sent_0*.*mask*_is_here.*sep+*
                MAPPING="{'0':'Non-toxic','1':'Toxic'}"
                ;;
        esac

        if [[ $LOAD_TEMPLATES = "true" ]]; then
            FILENAME=$TEMPLATE_DIR/${TASK}/$K-${SEED}.sort.txt
            for TEMPLATE in $(head -n $NUM_TEMPLATES $FILENAME); do
                if [[ "$TEMPLATE" == *mask*sent_0* ]]; then
                    MASK_FIRST=true
                else
                    MASK_FIRST=false
                fi
                if [ "$MASK_FIRST" = true ]; then
                    python tools/generate_labels.py \
                       --overwrite_output_dir \
                       --output_dir /tmp/output \
                       --model_name_or_path $MODEL_NAME \
                       --output_file $OUTPUT_DIR/$TASK/$K-$SEED.txt \
                       --append_output_file \
                       --write_template \
                       --template $TEMPLATE \
                       --mapping $MAPPING \
                       --task_name $TASK \
                       --data_dir $DATA_DIR/$TASK/$K-$SEED \
                       --k_likely $K_LIKELY \
                       --k_neighbors $K_NEIGHBORS \
                       --n_pairs $(($N_PAIRS / $NUM_TEMPLATES)) \
                       --use_full_length \
                       --first_sent_limit 64 \
                       --per_device_eval_batch_size 16 \
                       $TASK_EXTRA
                else
                    python tools/generate_labels.py \
                       --overwrite_output_dir \
                       --output_dir /tmp/output \
                       --model_name_or_path $MODEL_NAME \
                       --output_file $OUTPUT_DIR/$TASK/$K-$SEED.txt \
                       --append_output_file \
                       --write_template \
                       --template $TEMPLATE \
                       --mapping $MAPPING \
                       --task_name $TASK \
                       --data_dir $DATA_DIR/$TASK/$K-$SEED \
                       --k_likely $K_LIKELY \
                       --k_neighbors $K_NEIGHBORS \
                       --n_pairs $(($N_PAIRS / $NUM_TEMPLATES)) \
                       --use_full_length \
                       --truncate_head \
                       --first_sent_limit 64 \
                       --per_device_eval_batch_size 16 \
                       $TASK_EXTRA
                fi
                
            done
        else
            if [ "$MASK_FIRST" = true ]; then
                python tools/generate_labels.py \
                   --overwrite_output_dir \
                   --output_dir /tmp/output \
                   --model_name_or_path $MODEL_NAME \
                   --output_file $OUTPUT_DIR/$TASK/$K-$SEED.txt \
                   --template $TEMPLATE \
                   --mapping $MAPPING \
                   --task_name $TASK \
                   --data_dir $DATA_DIR/$TASK/$K-$SEED \
                   --k_likely $K_LIKELY \
                   --k_neighbors $K_NEIGHBORS \
                   --n_pairs $N_PAIRS \
                   --use_full_length \
                   --first_sent_limit 64 \
                   --per_device_eval_batch_size 16 \
                   $TASK_EXTRA
            else
                python tools/generate_labels.py \
                   --overwrite_output_dir \
                   --output_dir /tmp/output \
                   --model_name_or_path $MODEL_NAME \
                   --output_file $OUTPUT_DIR//manual_template/$TASK/$K-$SEED.txt \
                   --template $TEMPLATE \
                   --mapping $MAPPING \
                   --task_name $TASK \
                   --data_dir $DATA_DIR/$TASK/$K-$SEED \
                   --k_likely $K_LIKELY \
                   --k_neighbors $K_NEIGHBORS \
                   --n_pairs $N_PAIRS \
                   --use_full_length \
                   --truncate_head \
                   --first_sent_limit 64 \
                   --per_device_eval_batch_size 16 \
                   $TASK_EXTRA
            fi
        fi
    done
done
