from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import pandas as pd
from dcase_framework.sketch_utils import create_model, VerySimpleGenerator, get_callbacks, get_learner_params
import numpy as np


def process_dataset_txt(txt_filename, dataset_path=None):
    def make_full_path(x): return os.path.join(dataset_path, 'audio', x)
    def make_list(x): return eval(x)
    df = pd.read_csv(os.path.join(dataset_path, txt_filename))
    df['path'] = df['path'].apply(make_full_path)
    df['label'] = df['label'].apply(make_list)
    return df


def main():
    dataset_path = '../../audio_databases/magnatagatune/'
    meta_file = 'index/meta.csv'
    train_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    eval_folds = [12]
    test_folds = [13, 14, 15]

    # read and process meta, train, eval files - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    columns = ['path', 'scene_label', 'code']
    meta_df = process_dataset_txt(meta_file, dataset_path=dataset_path)

    train_df = meta_df[meta_df['fold'].isin(train_folds)]
    # eval_df = meta_df[meta_df['fold'].isin(test_folds)]

    # create training and evaluation generators  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    train_vsg = VerySimpleGenerator(train_df,
                                    batch_size=10,
                                    mono=True,
                                    desired_fs=16000,
                                    segment=True,
                                    frame_size_sec0=3.0,
                                    normalize=False,
                                    label_str='label')

    # iterate - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for idx, output in enumerate(train_vsg.flow()):
        print('{}/{} - {}, {}'.format(idx, train_vsg.get_num_batches(), np.shape(output[0]), np.shape(output[1])))
        del output



if __name__ == '__main__':
    main()
