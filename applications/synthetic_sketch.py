from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

from dcase_framework.sketch_utils import create_model, get_callbacks, get_learner_params
from dcase_framework.raw_audio_utils import SyntheticDataset


def main():
    parameter_set = 'dieleman2014'
    filename = './system/synthetic_sketch/{}'.format(parameter_set)  # TODO: create filename with hash

    params_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),'parameters','task1'+'.defaults.yaml')

    learner_params = get_learner_params(params_filename, parameter_set)

    # read and process meta, train, eval files - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ['sine' 'sawtooth' 'square' 'tri' 'whiteNoise' 'sineClip']

    synthetic_params = {'dataset_size': 3000,
                        'duration_sec': 3.0,
                        'fs': 16000,
                        'batch_size': 10,
                        'class_labels': ['sine', 'whiteNoise', 'square', 'sawtooth']}

    train_vsg = SyntheticDataset(**synthetic_params)
    eval_vsg = SyntheticDataset(**synthetic_params)

    # create model and callbacks list - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    model = create_model(learner_params, train_vsg.get_item_shape()[1:])

    callbacks = get_callbacks(filename, learner_params)

    model.fit_generator(train_vsg.flow(),
                        10,#train_vsg.get_num_batches(),
                        epochs=learner_params['training']['epochs'],
                        callbacks=callbacks,
                        validation_data=eval_vsg.flow() if learner_params['validation']['enable'] else None,
                        validation_steps=1)#eval_vsg.get_num_batches())


if __name__ == '__main__':
    main()
