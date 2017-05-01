import os
import numpy
import audioread
import librosa


__available_post_processing_methods__ = {}


def load_audio(filename, sr, mono):
    # output is (1, n_samples, n_channels)
    if os.path.isfile(filename):
        if os.path.getsize(filename) > 0:

            # file info
            af_info = audioread.audio_open(filename)
            n_channels = af_info.channels if not mono else 1
            duration_sec = af_info.duration
            duration_smp = int(duration_sec * sr)
            duration_smp = int(numpy.ceil(duration_smp / sr - 1 / sr) * sr)

            # load audio
            x, fs = librosa.core.load(filename, sr=sr, mono=mono)

            x = librosa.util.fix_length(x, duration_smp)
            x = x.reshape((n_channels, duration_smp, 1)).T

        else:
            print('\n\nSize of file {} is {}.\n'.format(os.path.basename(filename), os.path.getsize(filename)))
            return None, None
    else:
        raise IOError('File not found {}'.format(filename))

    return x, fs


def segment_audio(x, sr, frame_size_sec, hop_size_sec=None, **kwargs):
    n_channels = x.shape[-1]
    n_samples = x.shape[1]
    frame_size_smp = int(frame_size_sec * sr)
    hop_size_smp = int(hop_size_sec * sr) if hop_size_sec is not None else frame_size_smp

    frame_list = []
    start = 0
    while start < n_samples - (frame_size_smp - hop_size_smp):
        end = int(numpy.min((start + frame_size_smp, n_samples)))
        frame = x[:, start:end, :]

        if frame.shape[1] < frame_size_smp:
            pad_matrix = numpy.zeros((1, frame_size_smp - frame.shape[1], n_channels))
            frame = numpy.concatenate((frame, pad_matrix), axis=1)

        frame_list.append(frame)

        start += hop_size_smp

    return numpy.array(frame_list)[:, 0, :]  # numpy.squeeze(frame_list, axis=1)


__available_post_processing_methods__['segment_audio'] = segment_audio


def normalize(x, **kwargs):
    n_segments = x.shape[0]
    n_channels = x.shape[-1]
    for segment in range(n_segments):
        norm_val = numpy.max(numpy.max(numpy.abs(x[segment]), axis=1 if n_channels == 2 else 0))
        x[segment] /= norm_val
    return x


__available_post_processing_methods__['normalize'] = normalize


def mel_spectrogram(src, sr=12000, n_fft=512, n_mels=96, hop_len=256, dura=29.12, **kwargs):

    src = src[0, :, 0].T

    n_sample = src.shape[0]
    n_sample_wanted = int(dura * sr)

    # trim the signal at the center
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = src[int((n_sample - n_sample_wanted) / 2):int((n_sample + n_sample_wanted) / 2)]

    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(
        melgram(y=src, sr=sr, hop_length=hop_len, n_fft=n_fft, n_mels=n_mels) ** 2,
        ref_power=1.0)

    x = numpy.expand_dims(numpy.expand_dims(x, axis=3), axis=0)
    return x


__available_post_processing_methods__['mel_spectrogram'] = mel_spectrogram


class VerySimpleGenerator():
    def __init__(self, files_df, batch_size=1, mono=True, desired_fs=22050,
                 shuffle=True, label_str='scene_label', post_processing_list=None):

        self.files_df = files_df  # pd.DataFrame: for training: columns=['path', 'label',..], for test ['path']
        self.n_files = len(self.files_df)

        self.shuffle = shuffle
        if self.shuffle:
            self.files_df = self.files_df.sample(frac=1).reset_index(drop=True)

        self.label_str = label_str
        if self.label_str in self.files_df.columns:
            # check if str label or already code
            item_label = self.files_df.iloc[numpy.random.randint(0, self.n_files)][self.label_str]
            if all(isinstance(item, int) and item in [0, 1] for item in item_label):
                self.label_already_formatted = True
            else:
                self.label_already_formatted = False
                self.class_labels = self.files_df[self.label_str].unique()
                self.n_classes = len(self.class_labels)
        else:
            print('{} was not found in df'.format(label_str))

        self.post_processing_list = post_processing_list

        self.batch_size = batch_size
        self.mono = mono
        self.desired_fs = desired_fs

        self.n_frames = None
        self.frame_size_smp = None
        self.n_channels = None
        self.duration_smp = None

        self.n_batches = None

        self.item_shape = self.get_item_shape()

    def get_num_batches(self):
        if self.n_batches is None:
            self.n_batches = int(numpy.ceil(len(self.files_df) / self.batch_size))
        return self.n_batches

    def get_item_shape(self):
        # get a random file in the data set
        f = self.files_df.iloc[numpy.random.randint(self.n_files)]['path']
        # return its shape
        return numpy.shape(self.get_item_data(f))

    def get_item_data(self, item_filename):

        item_data, sr = load_audio(item_filename, self.desired_fs, self.mono)

        if item_data is None:
            return numpy.zeros(self.get_item_shape())

        if self.post_processing_list is not None:

            if type(self.post_processing_list) is not list:
                self.post_processing_list = [self.post_processing_list]

            for postproc_stage in self.post_processing_list:
                for method_name, params in postproc_stage.items():
                    if method_name in __available_post_processing_methods__.keys():
                        method = __available_post_processing_methods__[method_name]

                        if 'sr' in method.__code__.co_varnames:
                            params['sr'] = self.desired_fs
                    else:
                        raise IOError('Method not available {}'.format(method_name))

                    if params['enable']:
                        item_data = method(item_data, **params)

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

        if self.shuffle:
            order = numpy.random.permutation(x_training.shape[0])
            x_training = x_training[order, :, :]
            y_training = y_training[order, :]

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



