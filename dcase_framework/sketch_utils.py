import os
import importlib
import copy
import numpy
import audioread
import librosa
from dcase_framework.containers import DottedDict
from dcase_framework.parameters import ParameterContainer


def get_learner_params(parameters_filename, parameter_set):

    # Initialize ParameterContainer
    params = ParameterContainer(project_base=os.path.dirname(os.path.realpath(__file__)))

    # Load default parameters from a file
    params.load(filename=parameters_filename)

    params['active_set'] = parameter_set

    # Process parameters
    params.process()

    return DottedDict(params.get_path('learner')['parameters'])


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


class VerySimpleGenerator():
    def __init__(self, files_df, batch_size=1, mono=True, desired_fs=22050, segment=True,
                 frame_size_sec0=5.0, normalize=False, shuffle=True, label_str='scene_label'):

        self.files_df = files_df # pd.DataFrame: for training: columns=['path', 'label',..], for test ['path']
        self.n_files = len(self.files_df)

        if shuffle:
            self.files_df = self.files_df.sample(frac=1).reset_index(drop=True)

        self.label_str = label_str
        if self.label_str in self.files_df.columns:
            # check if str label or already code
            item_label = self.files_df.iloc[numpy.random.randint(0,self.n_files)][self.label_str]
            if all(isinstance(item, int) and item in [0, 1] for item in item_label):
                self.label_already_formatted = True
            else:
                self.label_already_formatted = False
                self.class_labels = self.files_df[self.label_str].unique()
                self.n_classes = len(self.class_labels)
        else:
            print('{} was not found in df'.format(label_str))

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

        if os.path.isfile(item_filename) and os.path.getsize(item_filename) > 0:
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
            if os.path.getsize(item_filename) == 0:
                print('\n\nSize of file {} is {}. Ignoring file.\n'.format(
                    os.path.basename(item_filename), os.path.getsize(item_filename)))
                return numpy.zeros((self.n_frames, self.frame_size_smp, self.n_channels))
            else:
                raise IOError("File not found [%s]" % (item['file']))

        return item_data

    def labels_to_matrix(self, data, labels):
        labels_one_hot = {}
        for item_filename, item_data in data.items():
            n_segments = item_data.shape[0]
            item_label = labels[item_filename]

            if self.label_already_formatted:
                labels_one_hot[item_filename] = numpy.tile(item_label, (n_segments, 1))
            else:
                pos = numpy.where(self.class_labels == item_label)
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




