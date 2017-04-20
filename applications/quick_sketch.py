from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy as np
import copy
import importlib
import pandas as pd

from dcase_framework.application_core import AcousticSceneClassificationAppCore
from dcase_framework.parameters import ParameterContainer
from dcase_framework.utils import *
from dcase_framework.containers import DottedDict
from dcase_framework.raw_audio_utils import SimpleGenerator


def learner_batcher_params(parameter_set):
    # Load default parameters from a file
    default_parameters_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parameters',
                                               'task1' + '.defaults.yaml')

    # Initialize ParameterContainer
    params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)))

    # Load default parameters from a file
    params.load(filename=default_parameters_filename)

    params['active_set'] = parameter_set

    # Process parameters
    params.process()

    learner_params = DottedDict(params.get_path('learner')['parameters'])

    return learner_params


def create_model(learner_params, input_shape):
    from keras.models import Sequential
    model = Sequential()
    model_params = copy.deepcopy(learner_params.get_path('model.config'))

    if 'conv_kernel' in learner_params['model']:
        kernel_params = learner_params['model']['conv_kernel']
        kernel = getattr(importlib.import_module("keras.initializers"), kernel_params['class'])
        kernel = kernel(**kernel_params['config'])
        set_kernel = True

    for layer_id, layer_setup in enumerate(model_params):
        layer_setup = DottedDict(layer_setup)
        try:
            LayerClass = getattr(importlib.import_module("keras.layers"), layer_setup['class_name'])
        except AttributeError:
            message = '{name}: Invalid Keras layer type [{type}].'.format(
                name=__class__.__name__,
                type=layer_setup['class_name']
            )
            raise AttributeError(message)

        if 'config' not in layer_setup:
            layer_setup['config'] = {}

        # Set layer input
        if layer_id == 0 and layer_setup.get_path('config.input_shape') is None:
            # Set input layer dimension for the first layer if not set
            if layer_setup.get('class_name') == 'Dropout':
                layer_setup['config']['input_shape'] = (input_shape,)
            else:
                layer_setup['config']['input_shape'] = input_shape

        elif layer_setup.get_path('config.input_dim') == 'FEATURE_VECTOR_LENGTH':
            layer_setup['config']['input_dim'] = input_shape

        # Set layer output
        if layer_setup.get_path('config.units') == 'CLASS_COUNT':
            layer_setup['config']['units'] = len(class_labels)

        # Set kernel initializer
        if 'kernel_initializer' in layer_setup.get('config') and set_kernel:
            layer_setup.get('config')['kernel_initializer'] = kernel

        if layer_setup.get('config'):
            model.add(LayerClass(**dict(layer_setup.get('config'))))
        else:
            model.add(LayerClass())

    try:
        OptimizerClass = getattr(importlib.import_module("keras.optimizers"),
                                 learner_params.get_path('model.optimizer.type')
                                 )

    except AttributeError:
        message = '{name}: Invalid Keras optimizer type [{type}].'.format(
            name=__class__.__name__,
            type=learner_params.get_path('model.optimizer.type')
        )
        raise AttributeError(message)

    model.compile(
        loss=learner_params.get_path('model.loss'),
        optimizer=OptimizerClass(**dict(learner_params.get_path('model.optimizer.parameters', {}))),
        metrics=learner_params.get_path('model.metrics')
    )

    return model


def main(argv):

    train_folds = [1, 2, 3, 4]

    # get params
    learner_params = learner_batcher_params('dieleman2014')

    # get dataset and create annotation
    # magnatagatune_audio_path = '/Users/JL/Documents/SMC10/Master-Thesis/Sound data bases/MagnatagatuneDataset/audio'
    magnatagatune_audio_path = '../../MagnatagatuneDataset/audio'
    # magnatagatune_meta = '/Users/JL/Documents/SMC10/Master-Thesis/Sound data bases/MagnatagatuneDataset/index/meta.csv'
    magnatagatune_meta = '../../MagnatagatuneDataset/index'
    magna_meta = pd.read_csv(magnatagatune_meta)

    meta_folds = magna_meta[magna_meta['fold'].isin(train_folds)]

    train_annotation = {}
    for idx, row in meta_folds.iterrows():
        fullpath_filename = os.path.join(magnatagatune_audio_path, row['path'])
        train_annotation[fullpath_filename] = {'file': fullpath_filename,
                                         'label': eval(row['label'])}

    # crate training and validation batch generators
    batch_size = learner_params.get_path('training.batch_size', 1)
    mono = learner_params.get_path('audio.mono', True)
    desired_fs = learner_params.get_path('audio.desired_fs', 22050)
    segment = False  #learner_params.get_path('audio.segment', True)
    frame_size_sec0 = learner_params.get_path('audio.frame_size_sec', 10.0)
    normalize = learner_params.get_path('audio.normalize', True)

    # create batcher
    sg = SimpleGenerator(train_annotation, batch_size, mono, desired_fs, segment, frame_size_sec0, normalize)
    print(sg.get_num_batches())
    # print(sg.get_item_shape())
    # print(next(sg.flow()))

    # create model
    input_shape = sg.get_item_shape()[1:]
    model = create_model(learner_params, input_shape)

    # train model
    model.fit_generator(sg.flow(), sg.get_num_batches(), 10)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)