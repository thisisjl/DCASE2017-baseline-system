from __future__ import print_function, absolute_import
import sys
import os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy
import pandas as pd
import audioread
import librosa
import importlib

from dcase_framework.parameters import ParameterContainer
from dcase_framework.containers import DottedDict
from dcase_framework.sketch_utils import create_model


def get_callbacks(filename, learner_params):

    callback_params = learner_params['training']['callbacks']
    callbacks = []
    if callback_params:
        for cp in callback_params:
            if cp['type'] == 'ModelCheckpoint' and not cp['parameters'].get('filepath'):
                cp['parameters']['filepath'] = os.path.splitext(filename)[
                                                   0] + '.weights.{epoch:02d}-{val_loss:.2f}.hdf5'

            if cp['type'] == 'EarlyStopping' and cp.get('parameters').get('monitor').startswith(
                    'val_') and not learner_params['validation'].get('enable', False):
                message = '{name}: Cannot use callback type [{type}] with monitor parameter [{monitor}] as there is no validation set.'.format(
                    name='',#self.__class__.__name__,
                    type=cp['type'],
                    monitor=cp.get('parameters').get('monitor')
                )

                # self.logger.exception(message)
                raise AttributeError(message)

            try:
                # Get Callback class
                CallbackClass = getattr(importlib.import_module("keras.callbacks"), cp['type'])

                # Add callback to list
                callbacks.append(CallbackClass(**cp.get('parameters', {})))

            except AttributeError:
                message = '{name}: Invalid Keras callback type [{type}]'.format(
                    name='',#self.__class__.__name__,
                    type=cp['type']
                )

                # self.logger.exception(message)
                raise AttributeError(message)

    return callbacks


def process_dataset_txt(txt_filename, columns=None, sep='\t', dataset_path=None):
    def make_full_path(x): return os.path.join(dataset_path, x)
    df = pd.read_csv(os.path.join(dataset_path, txt_filename), sep=sep)
    df.columns = columns
    df['path'] = df['path'].apply(make_full_path)
    return df


def get_learner_params(parameter_set):
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

    return DottedDict(params.get_path('learner')['parameters'])


class VerySimpleGenerator():
    def __init__(self, files_df, batch_size=1, mono=True, desired_fs=22050, segment=True,
                 frame_size_sec0=5.0, normalize=False, shuffle=True, label_str='scene_label'):

        self.files_df = files_df # pd.DataFrame: for training: columns=['path', 'label',..], for test ['path']
        self.n_files = len(self.files_df)

        if shuffle:
            self.files_df = self.files_df.sample(frac=1).reset_index(drop=True)

        self.label_str = label_str
        if label_str in self.files_df.columns:
            self.class_labels = self.files_df['scene_label'].unique()
        else:
            print('{} was not found in df'.format(label_str))

        self.n_classes = len(self.class_labels)

        self.batch_size = batch_size
        self.mono = mono
        self.desired_fs = desired_fs
        self.segment = segment
        self.frame_size_sec0 = frame_size_sec0
        self.normalize = normalize
        self.frame_size_smp0 = int(frame_size_sec0 * desired_fs)

        self.n_frames = None
        self.frame_size_smp = None
        self.n_channels = None
        self.duration_smp = None

        self.n_batches = None

        self.item_shape = self.get_item_shape()

    def get_num_batches(self):
        if self.n_batches is None:
            self.n_batches = int(numpy.ceil(len(self.files_df)/self.batch_size))
        return self.n_batches

    def get_item_shape(self):

        # get a random file in the data set
        f = self.files_df.iloc[numpy.random.randint(self.n_files)]['path']

        # audio info
        af_info = audioread.audio_open(f)
        self.n_channels = af_info.channels if not self.mono else 1
        duration_sec = af_info.duration
        self.duration_smp = int(duration_sec * self.desired_fs)
        self.duration_smp = int(numpy.ceil(self.duration_smp / self.desired_fs - 1 / self.desired_fs) * self.desired_fs)

        if self.segment:
            # compute number of frames
            self.n_frames = int(numpy.ceil(self.duration_smp / self.frame_size_smp0))
            # compute final duration of each frame
            self.frame_size_smp = int(self.duration_smp / self.n_frames)


        # return its shape
        return numpy.shape(self.get_item_data(f))

    def create_segments(self, audio):
        n_channels = numpy.shape(audio)[-1]
        start = 0
        end = self.frame_size_smp

        frame_matrix = numpy.zeros((self.n_frames, self.frame_size_smp, n_channels))

        for f_idx in range(self.n_frames):
            frame_matrix[f_idx, :, :] = audio[:, start:end, :]

            start += self.frame_size_smp
            end = start + self.frame_size_smp

        return frame_matrix

    def get_item_data(self, item_filename):

        if os.path.isfile(item_filename):
            item_data0, fs = librosa.core.load(item_filename, sr=self.desired_fs, mono=self.mono)
            item_data0 = librosa.util.fix_length(item_data0, self.duration_smp)

            item_data = item_data0.reshape((self.n_channels, self.duration_smp, 1)).T
            # item_data = item_data[:, :int(numpy.ceil(self.duration_smp / fs - 1 / fs) * fs), :]

            if self.segment:
                # TODO: segment with hop_size
                item_data = self.create_segments(item_data)

            if self.normalize:
                n_segments = item_data.shape[0]
                n_channels = item_data.shape[-1]
                for segment in range(n_segments):
                    norm_val = numpy.max(numpy.max(numpy.abs(item_data[segment]), axis=1 if n_channels == 2 else 0))
                    item_data[segment] /= norm_val
        else:
            raise IOError("File not found [%s]" % (item['file']))

        return item_data

    def labels_to_matrix(self, data, labels):
        labels_one_hot = {}
        for item_filename, item_data in data.items():
            n_segments = item_data.shape[0]
            pos = numpy.where(self.class_labels == labels[item_filename])
            roll = numpy.zeros((n_segments, self.n_classes))
            roll[:, pos] = 1

            labels_one_hot[item_filename] = roll

        return labels_one_hot

    def reset_output_arrays(self):
        self.batch_files = []
        self.batch_data = {}
        self.batch_labels = {}
        pass

    def process_output(self):

        # Convert annotations into activity matrix format
        labels_one_hot = self.labels_to_matrix(data=self.batch_data, labels=self.batch_labels)

        x_training = numpy.vstack([self.batch_data[x] for x in self.batch_files])
        y_training = numpy.vstack([labels_one_hot[x] for x in self.batch_files])

        return x_training, y_training

    def flow(self):
        # sequence = annotation.keys()

        while True:
            batch_idx = 0

            # for item_filename in self.sequence:

            for idx, item in self.files_df.iterrows():
                item_filename = item['path']
                label = item[self.label_str]

                if batch_idx == 0:
                    self.reset_output_arrays()

                self.batch_files.append(item_filename)
                self.batch_labels[item_filename] = label
                self.batch_data[item_filename] = self.get_item_data(item_filename)

                if batch_idx == self.batch_size - 1:

                    batch_idx = 0  # reinitialize batch counter

                    # output of generator
                    x_training, y_training = self.process_output()
                    yield x_training, y_training

                else:
                    batch_idx += 1

            if not batch_idx == 0:
                # output of generator
                x_training, y_training = self.process_output()
                yield x_training, y_training


def main():
    parameter_set = 'dieleman2014'
    dcase2017_path = '../../../Sound data bases/DCASE 2017/TUT-acoustic-scenes-2017-development'
    evaluation_setup = 'evaluation_setup'
    meta_file = 'meta.txt'
    fold = 1
    filename = '/system/dcase_scenes_sketch/{}'.format(parameter_set)  # TODO: create filename with hash

    learner_params = get_learner_params(parameter_set)

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
