import pandas as pd
import os
import argparse
import numpy as np
import traceback
from tqdm import tqdm
from transformers import AutoTokenizer
import re
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def truncate_text(text):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Truncate the sequence to a maximum length of 128 tokens
    max_length = 128
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        tokens[-1] = tokenizer.sep_token

    # Convert the tokens back to text
    truncated_text = tokenizer.convert_tokens_to_string(tokens)

    return truncated_text

def preprocess_sentence(sentence):
    # Remove tags
    sentence = re.sub(r'</s>|==', '', sentence)
    # Remove quotes
    sentence = sentence.replace('"', '').replace("'", "")
    # Remove URLs
    sentence = re.sub(r'http\S+|www\S+', '', sentence)
    # Remove < > tags
    sentence = re.sub(r'<.*?>', '', sentence)
    return sentence.strip()



parser = argparse.ArgumentParser(description='create k-shot for template and label generation')
parser.add_argument('--lang', type=str, nargs='+', default=['en', 'es', 'fr'], help="add languages") # change
parser.add_argument('--toxic_data_dir', type=str, default='jigsaw/processed')
parser.add_argument('--lmbff-dir', type=str, default='LM-BFF/data/')
parser.add_argument('--k', type=int, default=16)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--file_ext', type=str, default='csv')
parser.add_argument('--label_first', action="store_true")
parser.add_argument('--include_header', action="store_true")

def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    total_examples = len(args.lang) * 2 * args.k
    test_examples = 0

    if args.label_first:
        train_df = pd.DataFrame(columns=['label', 'sentence'])
        dev_df = pd.DataFrame(columns=['label', 'sentence'])
        test_df = pd.DataFrame(columns=['label', 'sentence'])
    else:    
        train_df = pd.DataFrame(columns=['sentence', 'label'])
        dev_df = pd.DataFrame(columns=['sentence', 'label'])
        test_df = pd.DataFrame(columns=['sentence', 'label'])

    for lang in tqdm(args.lang):
        lang_files = [file for file in os.listdir(os.path.join(args.toxic_data_dir)) if file.__contains__("_"+lang)]
        print(lang_files)
        for file in lang_files:
            if file.__contains__('test'):
                test_data = pd.read_csv(os.path.join(args.toxic_data_dir, file), usecols=['sentence', 'label'])
                test_examples += test_data.shape[0]
                test_data['sentence'] = test_data['sentence'].apply(lambda x: truncate_text(preprocess_sentence(x)))
                test_df = pd.concat([test_df, test_data], axis=0)
            else:
                data = pd.read_csv(os.path.join(args.toxic_data_dir, file), usecols=['sentence', 'label'])
                instances = data.values
                np.random.shuffle(instances)

                # get list of sentences per label
                label_list = {}
                for sent, label in instances:
                    if label not in label_list:
                        label_list[label] = [sent]
                    else:
                        label_list[label].append(sent)
                
                assert len(label_list.keys()) == 2

                # sample top-k
                for label in label_list:
                    if file.__contains__('train'):
                        train_df = train_df.append(list(map(lambda x: {'sentence': truncate_text(preprocess_sentence(x)), 'label': label}, label_list[label][:args.k])), ignore_index=True)
                        
                    elif file.__contains__('dev'):
                        dev_df = dev_df.append(list(map(lambda x: {'sentence': truncate_text(preprocess_sentence(x)), 'label': label}, label_list[label][:args.k])), ignore_index=True)
                    else:
                        raise ValueError(f'unknown file type: {file}')
                        
    assert len(train_df) == total_examples, f'expected {total_examples}, but sampled {len(train_df)}'
    assert len(dev_df) == total_examples, f'expected {total_examples}, but sampled {len(dev_df)}'
    assert len(test_df) == test_examples, f'expected {test_examples}, but sampled {len(test_df)}'

    # create save directory for LM-BFF
    save_dir = os.path.join(
        args.lmbff_dir,
        'k-shot', 'toxic', f'{args.k * len(args.lang)}-{args.seed}'
    )
    os.makedirs(save_dir, exist_ok=True)

    # write file
    column_order = ['label', 'sentence'] if args.label_first else ['sentence', 'label']
    print(f'Writing to {save_dir} with column order {column_order}')
    
    train_df.to_csv(os.path.join(save_dir, f'train.{args.file_ext}'), 
                        columns=column_order, 
                        sep='\t' if args.file_ext == 'tsv' else ',', 
                        index=False, 
                        header=args.include_header
                    )

    dev_df.to_csv(os.path.join(save_dir, f'dev.{args.file_ext}'), 
                        columns=column_order, 
                        sep='\t' if args.file_ext == 'tsv' else ',', 
                        index=False, 
                        header=args.include_header
                    )

    test_df.to_csv(os.path.join(save_dir, f'test.{args.file_ext}'), 
                        columns=column_order, 
                        sep='\t' if args.file_ext == 'tsv' else ',', 
                        index=False, 
                        header=args.include_header
                    )
    
if __name__ == '__main__':
    main()
