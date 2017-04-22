from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import pandas as pd
from dcase_framework.sketch_utils import create_model, VerySimpleGenerator, get_callbacks, get_learner_params


def process_dataset_txt(txt_filename, columns=None, sep='\t', dataset_path=None):
    def make_full_path(x): return os.path.join(dataset_path, x)
    df = pd.read_csv(os.path.join(dataset_path, txt_filename), sep=sep)
    df.columns = columns
    df['path'] = df['path'].apply(make_full_path)
    return df


def main():
    parameter_set = 'dieleman2014'
    dcase2017_path = '../../../Sound data bases/DCASE 2017/TUT-acoustic-scenes-2017-development'
    evaluation_setup = 'evaluation_setup'
    meta_file = 'meta.txt'
    fold = 1
    filename = '/system/dcase_scenes_sketch/{}'.format(parameter_set)  # TODO: create filename with hash

    params_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'parameters','task1'+'.defaults.yaml')

    learner_params = get_learner_params(params_filename, parameter_set)

    # read and process meta, train, eval files - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    columns = ['path', 'scene_label', 'code']
    meta_df = process_dataset_txt(meta_file, columns, dataset_path=dcase2017_path)

    columns = ['path', 'scene_label']
    train_df = process_dataset_txt(
        os.path.join(evaluation_setup, 'fold{}_train.txt'.format(fold)), columns, dataset_path=dcase2017_path)

    eval_df = process_dataset_txt(
        os.path.join(evaluation_setup, 'fold{}_evaluate.txt'.format(fold)), columns, dataset_path=dcase2017_path)

    # create training and evaluation generators  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    train_vsg = VerySimpleGenerator(train_df,
                                    batch_size=learner_params['training']['batch_size'],
                                    mono=learner_params['audio']['mono'],
                                    desired_fs=learner_params['audio']['desired_fs'],
                                    segment=learner_params['audio']['segment'],
                                    frame_size_sec0=learner_params['audio']['frame_size_sec'],
                                    normalize=learner_params['audio']['normalize'])

    eval_vsg = VerySimpleGenerator(eval_df,
                                   batch_size=learner_params['training']['batch_size'],
                                   mono=learner_params['audio']['mono'],
                                   desired_fs=learner_params['audio']['desired_fs'],
                                   segment=learner_params['audio']['segment'],
                                   frame_size_sec0=learner_params['audio']['frame_size_sec'],
                                   normalize=learner_params['audio']['normalize'])

    # create model and callbacks list - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model = create_model(learner_params, train_vsg.get_item_shape()[1:])

    callbacks = get_callbacks(filename, learner_params)

    model.fit_generator(train_vsg.flow(),
                        train_vsg.get_num_batches(),
                        epochs=learner_params['training']['epochs'],
                        callbacks=callbacks,
                        validation_data=eval_vsg.flow() if learner_params['validation']['enable'] else None,
                        validation_steps=eval_vsg.get_num_batches())


if __name__ == '__main__':
    main()
