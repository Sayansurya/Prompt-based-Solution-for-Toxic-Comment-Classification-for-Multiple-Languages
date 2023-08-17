'''
    We want to keep only SENTENCE - LABEL in the .tsv file
'''

import pandas as pd
import os
import argparse
import numpy as np
import emoji
import traceback

parser = argparse.ArgumentParser(description='create k-shot data for template and label generation')

parser.add_argument('--lang', type=str, nargs='+', default=['eng', 'hin', 'spanish', 'mal', 'tam'], help="add languages")
parser.add_argument('--task', type=str, default='a', help='choose task')
parser.add_argument('--toxic_data_dir', type=str, default='dataset/processed')
parser.add_argument('--lmbff-dir', type=str, default='LM-BFF/data/')
parser.add_argument('--k', type=int, default=16)
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--file_ext', type=str, default='csv')
parser.add_argument('--label_first', action="store_true")
parser.add_argument('--include_header', action="store_true")

def main():
    args = parser.parse_args()
    print(args, type(args.label_first))
    np.random.seed(args.seed)    
    if args.task.lower() == 'a':
        total_examples = len(args.lang) * 3 * args.k 
        test_examples = 0
        if args.label_first:
            train_df = pd.DataFrame(columns=['label', 'sentence'])
            dev_df = pd.DataFrame(columns=['label', 'sentence'])
            test_df = pd.DataFrame(columns=['label', 'sentence'])
        else:    
            train_df = pd.DataFrame(columns=['sentence', 'label'])
            dev_df = pd.DataFrame(columns=['sentence', 'label'])
            test_df = pd.DataFrame(columns=['sentence', 'label'])

        for lang in args.lang:
            files = [file for file in os.listdir(os.path.join(args.toxic_data_dir, 'taskA')) if file.__contains__(lang)] 

            for file in files:
                if file.__contains__('train'):
                    dset_type='train'
                    df = train_df
                else:
                    dset_type='dev'
                    df = dev_df

                data = pd.read_csv(os.path.join(args.toxic_data_dir, 'taskA', file), usecols=['text', 'label'])
                if dset_type == 'dev': test_examples += data.shape[0]
                instances = data.values
                np.random.shuffle(instances)
                
                # create save directory for LM-BFF
                save_dir = os.path.join(
                    args.lmbff_dir,
                    'k-shot', 'toxic', f'{args.k * len(args.lang)}-{args.seed}'
                )
                os.makedirs(save_dir, exist_ok=True)

                # get list of sentences per label
                label_list = {} 
                for sent, label in instances:
                    if label not in label_list:
                        label_list[label] = [emoji.demojize(sent.strip().replace('\n'))]
                    else:
                        label_list[label].append(emoji.demojize(sent.strip().replace('\n')))

                # check if per-class examples are less than k
                for label in label_list:
                    if len(label_list[label]) < args.k:
                        try:
                            augment = pd.read_csv(f'dataset/augment/{lang}_{dset_type}.csv', usecols=['sentence', 'label'])
                            augment_instances = augment.values
                            for sent, augment_label in augment_instances:
                                # if lang == 'hin':
                                #         print(len(label_list[0]), len(label_list[1]), len(label_list[2]))
                                #         print(lang, dset_type, augment_label, label, type(augment_label), type(label))
                                if augment_label == label:
                                    label_list[label].append(emoji.demojize(sent.strip().replace('\n')))
                                if len(label_list[label]) >= args.k: break
                        except:
                            traceback.print_exc()
                            raise NotImplementedError(f'Case for {dset_type}-{lang} not implemented.')
     
                # # create df of filtered k-shot instances per label
                # mapper = {k:str(v) for v,k in enumerate(sorted(set(label_list)))}
                # print({v:k for k,v in mapper.items()})

                for label in label_list:
                    print(f'for {file}, {label} added:- {len(label_list[label][:args.k])} examples')
                    for instance in label_list[label][:args.k]:
                        if args.label_first:
                            df.loc[len(df)] = {'label': label, 'sentence': instance}
                        else:
                            df.loc[len(df)] = {'sentence': instance, 'label': label}

                if dset_type == 'dev':
                    data['text'] = data['text'].apply(lambda x: emoji.demojize(x.strip().replace('\n')))
                    test_df = pd.concat([test_df, data.rename(columns={'text': 'sentence'})])

        assert train_df.shape[0] == total_examples
        assert dev_df.shape[0] == total_examples
        assert test_df.shape[0] == test_examples

        column_order = ['label', 'sentence'] if args.label_first else ['sentence', 'label']
        print(f'Writing to {save_dir} with column order {column_order}')
        train_df.to_csv(os.path.join(save_dir, f'train.{args.file_ext}'), columns=column_order, sep='\t' if args.file_ext == 'tsv' else ',', index=False, header=args.include_header)
        dev_df.to_csv(os.path.join(save_dir, f'dev.{args.file_ext}'), columns=column_order, sep='\t' if args.file_ext == 'tsv' else ',', index=False, header=args.include_header)
        test_df.to_csv(os.path.join(save_dir, f'test.{args.file_ext}'), columns=column_order, sep='\t' if args.file_ext == 'tsv' else ',', index=False, header=args.include_header)
            
if __name__ == '__main__':
    main()