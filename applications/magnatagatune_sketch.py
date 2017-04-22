from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import pandas as pd
from dcase_framework.sketch_utils import create_model, VerySimpleGenerator, get_callbacks, get_learner_params


def process_dataset_txt(txt_filename, dataset_path=None):
    def make_full_path(x): return os.path.join(dataset_path, 'audio', x)
    def make_list(x): return eval(x)
    df = pd.read_csv(os.path.join(dataset_path, txt_filename))
    df['path'] = df['path'].apply(make_full_path)
    df['label'] = df['label'].apply(make_list)
    return df


def main():
    parameter_set = 'dieleman2014'
    dataset_path = '../../../Sound data bases/MagnatagatuneDataset/'
    evaluation_setup = 'evaluation_setup'
    meta_file = 'index/meta.csv'
    fold = 1
    filename = './system/magnatagatune_sketch/{}'.format(parameter_set)  # TODO: create filename with hash
    train_folds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    eval_folds = [12]
    test_folds = [13, 14, 15]

    params_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameters', 'task1'+'.defaults.yaml')

    learner_params = get_learner_params(params_filename, parameter_set)

    # read and process meta, train, eval files - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    columns = ['path', 'scene_label', 'code']
    meta_df = process_dataset_txt(meta_file, dataset_path=dataset_path)

    train_df = meta_df[meta_df['fold'].isin(train_folds)]
    eval_df = meta_df[meta_df['fold'].isin(test_folds)]

    # create training and evaluation generators  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    train_vsg = VerySimpleGenerator(train_df,
                                    batch_size=learner_params['training']['batch_size'],
                                    mono=learner_params['audio']['mono'],
                                    desired_fs=learner_params['audio']['desired_fs'],
                                    segment=learner_params['audio']['segment'],
                                    frame_size_sec0=learner_params['audio']['frame_size_sec'],
                                    normalize=learner_params['audio']['normalize'],
                                    label_str='label')

    eval_vsg = VerySimpleGenerator(eval_df,
                                   batch_size=learner_params['training']['batch_size'],
                                   mono=learner_params['audio']['mono'],
                                   desired_fs=learner_params['audio']['desired_fs'],
                                   segment=learner_params['audio']['segment'],
                                   frame_size_sec0=learner_params['audio']['frame_size_sec'],
                                   normalize=learner_params['audio']['normalize'],
                                   label_str='label')

    # create model and callbacks list - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model = create_model(learner_params, train_vsg.get_item_shape()[1:])

    callbacks = get_callbacks(filename, learner_params)

    model.fit_generator(train_vsg.flow(),
                        1,#train_vsg.get_num_batches(),
                        epochs=learner_params['training']['epochs'],
                        callbacks=callbacks,
                        validation_data=eval_vsg.flow() if learner_params['validation']['enable'] else None,
                        validation_steps=1)#eval_vsg.get_num_batches())


if __name__ == '__main__':
    main()
