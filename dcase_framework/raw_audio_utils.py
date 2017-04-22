import os
import copy
import numpy
import importlib
import muda
import jams
import audioread
import librosa
from bisect import bisect
from random import random, uniform
from .files import AudioFile
from .features import FeatureContainer


class RawAudioBatcher():
    def __init__(self, split_files, _annotations, class_labels, batch_size=1, mono=True, desired_fs=22050,
                 segment=True, frame_size_sec0=5.0, normalize=False, augmentation_params=None):
        """
        RawAudioBatcher allows to create batches from audio data.
        It contains a generator method that can be used as input to the method fit_generator of a
        Keras neural network model with audio waveforms.

        Parameters
        ----------
        split_files : list
            Full list of audio files to load
        _annotations : dict
            Dictionary containing a nested dictionary for each audio file in split_files.
            The nested dictionary must contain the keys 'file', 'identifier' and'scene_label'. Example:
            {'a001_140_150.wav': {'file':'a001_140_150.wav', 'identifier': 'a001', 'scene_label': 'residential_area'}

        class_labels : list
            All possible class labels
        batch_size : int
            Number of audio files to load and output
        mono : bool
            If True the audio file will be mixed down to a mono file
        desired_fs : int
            Sampling frequency of the output data
        segment : bool
            Separate the audio file into segments. Its duration defined by frame_size_sec0.
        frame_size_sec0 : float
            number of seconds of each segment if segment is True
        normalize : bool
            Normalize output values between 0 and 1. If mono is False, channels are normalized with the same value.
        """
        self.files = split_files
        self.n_files = len(self.files)
        self.annotations = _annotations
        if self.n_files < batch_size:
            self.batch_size = self.n_files
        else:
            self.batch_size = batch_size
        self.mono = mono
        self.desired_fs = desired_fs
        self.normalize = normalize
        self.segment = segment
        self.frame_size_smp0 = int(frame_size_sec0 * desired_fs)

        self.generator_sequence = copy.copy(self.files)

        self.n_frames = None
        self.frame_size_smp = None
        self.n_channels = None
        self.duration_smp = None
        self.class_labels = class_labels

        self.item_shape = self.get_item_shape()

        self.batch_files = []
        self.batch_data = {}
        self.batch_annotations = {}

        self.n_batches = None

        # data augmentation parameters
        self.enable_augmentation = getattr(augmentation_params, 'enable', False)
        if self.enable_augmentation:
            self.augmentation_samples = None
            self.augmentation_chain = self.create_augmentation_chain(augmentation_params)
            # self.batch_size *= self.batch_size + self.batch_size * self.augmentation_samples

    def get_item_shape(self):
        for f in self.files:
            af_info = AudioFile(filename=f).info
            self.n_channels = af_info.channels if not self.mono else 1
            duration_sec = af_info.duration
            # fs = af_info.samplerate
            self.duration_smp = int(duration_sec * self.desired_fs)
            self.duration_smp = int(numpy.ceil(self.duration_smp / self.desired_fs - 1 / self.desired_fs) * self.desired_fs)
            break  # TODO: check if all files have same duration

        if self.segment:

            # compute number of frames
            self.n_frames = int(numpy.ceil(self.duration_smp / self.frame_size_smp0))

            # compute final duration of each frame
            self.frame_size_smp = int(self.duration_smp / self.n_frames)

            return self.n_frames, self.frame_size_smp, self.n_channels

        else:
            return 1, self.duration_smp, self.n_channels

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

    def generator(self):
        # split_files = self.files
        _annotations = self.annotations

        while True:
            batch_idx = 0

            for item_filename in self.generator_sequence:

                if batch_idx == 0:
                    self.reset_output_arrays()

                self.batch_files.append(item_filename)
                self.batch_annotations[item_filename] = _annotations[item_filename]

                item_data = self.get_item_data(item_filename)

                self.batch_data[item_filename] = item_data

                if batch_idx == self.batch_size - 1:

                    batch_idx = 0                                                       # reinitialize batch counter

                    # output of generator
                    x_training, y_training = self.process_output()
                    yield x_training, y_training

                else:
                    batch_idx += 1

            if not batch_idx == 0:
                # output of generator
                x_training, y_training = self.process_output()
                yield x_training, y_training

    def _get_target_matrix_dict(self, data, annotations):
        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            frame_count = data[audio_filename].feat[0].shape[0]
            pos = self.class_labels.index(annotations[audio_filename]['scene_label'])
            roll = numpy.zeros((frame_count, len(self.class_labels)))
            roll[:, pos] = 1
            activity_matrix_dict[audio_filename] = roll
        return activity_matrix_dict

    def frame_generator(self, filename, desired_fs=44100, frame_len_sec=5.0, hop_size_sec=2.5, mono=False):

        if os.path.isfile(filename):
            af = AudioFile(filename=filename)

            duration_sec = numpy.ceil(af.info.duration - 1 / desired_fs)

            n_channels = 1 if mono else af.info.channels

            if frame_len_sec > duration_sec:
                frame_len_sec = duration_sec
            frame_len_smp = int(frame_len_sec * desired_fs)

            if hop_size_sec is None:
                hop_size_sec = frame_len_sec

            start_sec = 0
            while start_sec < duration_sec - (frame_len_sec - hop_size_sec):
                stop_sec = (start_sec + frame_len_sec)
                item_data, _ = AudioFile().load(filename, fs=desired_fs, mono=mono, start=start_sec, stop=stop_sec)

                item_data = item_data.reshape((n_channels, frame_len_smp, 1)).T
                yield item_data

                start_sec += hop_size_sec

        else:
            print('{} not found'.format(item_filename))

    def get_item_data(self, item_filename):

        if os.path.isfile(item_filename):
            af = AudioFile(filename=item_filename)
            item_data_len = int(numpy.ceil(af.info.duration * self.desired_fs))

            item_data0, fs = AudioFile().load(item_filename, fs=self.desired_fs, mono=self.mono)

            item_data = item_data0.reshape((self.n_channels, item_data_len, 1)).T
            item_data = item_data[:, :int(numpy.ceil(self.duration_smp / fs - 1 / fs) * fs), :]

            if self.segment:
                # TODO: segment with hop_size
                item_data = self.create_segments(item_data)

            if self.normalize:
                n_segments = item_data.shape[0]
                n_channels = item_data.shape[-1]
                for segment in range(n_segments):
                    norm_val = numpy.max(numpy.max(numpy.abs(item_data[segment]), axis=1 if n_channels == 2 else 0))
                    item_data[segment] /= norm_val

            if self.enable_augmentation:
                item_data = self.do_augmentation(item_data)

            fc = FeatureContainer()
            fc.feat = [item_data]
        else:
            raise IOError("File not found [%s]" % (item['file']))

        return fc

    def create_batch(self, batch_size=None, return_item_name=False):

        if batch_size is not None:
            self.batch_size = batch_size

        self.reset_output_arrays()

        # for item_filename in order[:self.batch_size]:
        for item_filename in self.files[:self.batch_size]:

            self.batch_files.append(item_filename)
            self.batch_annotations[item_filename] = self.annotations[item_filename]
            self.batch_data[item_filename] = self.get_item_data(item_filename)

        # output of generator
        x_training, y_training = self.process_output()

        if return_item_name:
            return x_training, y_training, self.batch_files
        else:
            return x_training, y_training

    def reset_output_arrays(self):
        self.batch_files = []
        self.batch_data = {}
        self.batch_annotations = {}
        pass

    def process_output(self):

        # Convert annotations into activity matrix format
        activity_matrix_dict = self._get_target_matrix_dict(data=self.batch_data, annotations=self.batch_annotations)

        x_training = numpy.vstack([self.batch_data[x].feat[0] for x in self.batch_files])
        y_training = numpy.vstack([activity_matrix_dict[x] for x in self.batch_files])

        return x_training, y_training

    def get_num_batches(self):
        if self.n_batches is None:
            self.n_batches = int(numpy.ceil(len(self.files)/self.batch_size))

        return self.n_batches

    def create_augmentation_chain(self, augmentation_params):

        deformers_list = []
        chain_type = augmentation_params['chain_type']
        chain_length = len(augmentation_params['config'])

        if chain_type == 'Pipeline':
            print('DATA AUGMENTATION WARNING: chain type muda.Pipeline not supported yet. Changing it to Union')
            chain_type = 'Union'

        self.augmentation_samples = 0
        augmentation_config = augmentation_params['config']
        for deformer_id, deformer_setup in enumerate(augmentation_config):
            # deformer_setup = DottedDict(deformer_setup)

            try:
                DeformerClass = getattr(importlib.import_module("muda.deformers"), deformer_setup['class_name'])
            except AttributeError:
                message = '{name}: Invalid muda deformer class [{type}].'.format(
                    name=self.__class__.__name__,
                    type=deformer_setup['class_name']
                )
                self.logger.exception(message)
                raise AttributeError(message)

            if deformer_setup.get('config'):
                deformer = DeformerClass(**dict(deformer_setup.get('config')))
            else:
                deformer = DeformerClass()

            # create deformer tuple (name, deformer)
            deformer_name = 'deformer_{}'.format(deformer_id)

            # add a bypass in the first deformer, so we get the original input as first augmented output
            if deformer_id == 0 or chain_type == 'Pipeline':
                deformer = muda.deformers.Bypass(deformer)

            deformers_list.append((deformer_name, deformer))

            # comput number of augmentation samples
            if 'n_samples' in deformer_setup['config']:
                self.augmentation_samples += deformer_setup['config']['n_samples']
            elif deformer_setup['class_name'] == 'DynamicRangeCompression':
                self.augmentation_samples += len(deformer_setup['config']['preset'])
            else:
                self.augmentation_samples += 1





        # compute number of created samples in augmentation
        # self.augmentation_samples = \
        #     numpy.sum([deformer_setup['config']['n_samples'] for deformer_setup in augmentation_params['config']])

        # if chain_type == 'Pipeline':
        #     self.augmentation_samples = numpy.sum([self.nCr(chain_length, group_size) for group_size in range(chain_length)])
        # elif chain_type == 'Union':
        #     pass

        return getattr(importlib.import_module("muda"), augmentation_params['chain_type'])(steps=deformers_list)

    def do_augmentation(self, item_data):

        n_augmentations = self.augmentation_samples
        n_segments = item_data.shape[0]
        sgmt_duration = item_data.shape[1]
        n_channels = item_data.shape[2]
        augmented_data = numpy.zeros((n_segments + n_augmentations, sgmt_duration, n_channels))

        # create a jam for each row and channel in item_data
        for s_idx, segment in enumerate(item_data):
            # jam = []
            # for channel in range(numpy.shape(segment)[-1]):
            for channel, sgmt_ch in enumerate(segment.T):

                duration_smp = len(sgmt_ch)
                duration_sec = duration_smp / self.desired_fs

                jam = jams.JAMS()
                jam.file_metadata.duration = duration_sec

                j_orig = muda.jam_pack(jam, _audio=dict(y=sgmt_ch, sr=self.desired_fs))

                for aug_idx, j_new in enumerate(self.augmentation_chain.transform(j_orig)):
                    # for k in j_new['sandbox']['muda']['history']: print(k['transformer']['__class__'])

                    # make sure output duration is the same as input
                    augmented_audio = j_new['sandbox']['muda']['_audio']['y']

                    len_diff = duration_smp - len(augmented_audio)
                    if len_diff > 0:
                        augmented_audio = numpy.concatenate((augmented_audio, numpy.zeros(len_diff)))
                    if len_diff < 0:
                        augmented_audio = augmented_audio[:len_diff]

                    augmented_data[s_idx+aug_idx, :, channel] = augmented_audio

        return augmented_data

    def nCr(self, n, r):
        import math
        f = math.factorial
        return f(n) // f(r) // f(n - r)


class SimpleGenerator(RawAudioBatcher):
    def __init__(self, annotation, batch_size=1, mono=True, desired_fs=22050,
                 segment=True, frame_size_sec0=5.0, normalize=False):
        self.annotation = annotation
        self.files = annotation.keys()
        self.sequence = annotation.keys()
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

    def get_item_shape(self):
        for f in self.files:
            # af_info = AudioFile(filename=f).info
            af_info = audioread.audio_open(f)
            self.n_channels = af_info.channels if not self.mono else 1
            duration_sec = af_info.duration
            # fs = af_info.samplerate
            self.duration_smp = int(duration_sec * self.desired_fs)
            self.duration_smp = int(
                numpy.ceil(self.duration_smp / self.desired_fs - 1 / self.desired_fs) * self.desired_fs)
            break  # TODO: check if all files have same duration

        if self.segment:

            # compute number of frames
            self.n_frames = int(numpy.ceil(self.duration_smp / self.frame_size_smp0))

            # compute final duration of each frame
            self.frame_size_smp = int(self.duration_smp / self.n_frames)

            return self.n_frames, self.frame_size_smp, self.n_channels

        else:
            return 1, self.duration_smp, self.n_channels

    def get_item_data(self, item_filename):

        if os.path.isfile(item_filename):
            # af = audioread.audio_open(item_filename)
            # item_data_len = int(numpy.ceil(af.duration * self.desired_fs))

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

            # if self.enable_augmentation:
            #     item_data = self.do_augmentation(item_data)

            fc = FeatureContainer()
            fc.feat = [item_data]
        else:
            raise IOError("File not found [%s]" % (item['file']))

        return fc

    def _get_target_matrix_dict(self, data, annotations):
        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            frame_count = data[audio_filename].feat[0].shape[0]
            if 'scene_label' in annotations[audio_filename]:
                pos = self.class_labels.index(annotations[audio_filename]['scene_label'])
                roll = numpy.zeros((frame_count, len(self.class_labels)))
                roll[:, pos] = 1
                activity_matrix_dict[audio_filename] = roll

            elif 'label' in annotations[audio_filename]:
                label = annotations[audio_filename]['label']
                if type(label) == list:
                    activity_matrix_dict[audio_filename] = label
                else:
                    print('SimpleGenerator._get_targt_matrix_dict - unsupported label type')
                    sys.exit()
            else:
                print('SimpleGenerator._get_targt_matrix_dict - could not find label in annotation')
                sys.exit()

        return activity_matrix_dict

    def flow(self):
        # sequence = annotation.keys()

        while True:
            batch_idx = 0

            for item_filename in self.sequence:

                if batch_idx == 0:
                    self.reset_output_arrays()

                self.batch_files.append(item_filename)
                self.batch_annotations[item_filename] = self.annotation[item_filename]

                item_data = self.get_item_data(item_filename)

                self.batch_data[item_filename] = item_data

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


class SyntheticDataset():
    def __init__(self, dataset_size, duration_sec, fs, batch_size=10, class_labels=None,
                 class_prob=None, clip_r=0.85, trunc_t0=.8, trunc_t1=.9, freq_range0=20, freq_range1=22050):

        self.dataset_size = 17 if dataset_size is None else dataset_size
        self.duration = duration_sec
        self.fs = fs
        self.batch_size = batch_size
        self.n = int(self.duration * self.fs)
        self.clip_r = clip_r
        assert trunc_t0 < trunc_t1, 'trunc_t0 must be lower than trunc_t1'
        self.trunc_t0 = trunc_t0
        self.trunc_t1 = trunc_t1

        # self.signal_types = numpy.array(
        #    ['sine', 'sawtooth', 'square', 'tri', 'whiteNoise', 'sineClip', 'sineTrunc'], dtype=numpy.object_)

        self.signal_types = numpy.array(
            ['sine', 'sawtooth', 'square', 'tri', 'whiteNoise', 'sineClip'], dtype=numpy.object_)

        # self.signal_types = ['sine', 'sawtooth', 'square', 'tri', 'whiteNoise', 'sineClip', 'sineTrunc']

        # a = numpy.array(['a','b'],dtype=numpy.object_)

        if class_labels is None:
            self.class_labels = self.signal_types
        else:
            for c in class_labels:
                assert c in self.signal_types, '{} not supported as signal type'.format(c)
            self.class_labels = numpy.array(class_labels, dtype=numpy.object_)

        if class_prob is None:
            class_prob = [1 / len(self.class_labels)] * len(self.class_labels)
        else:
            assert numpy.sum(class_prob) == 1, 'class_prob should sum to one'

        # create sequence - - - - - - - - - - - - - - -

        # create a cumulative probability distribution for classes
        cdf = [class_prob[0]]
        for i in range(1, len(class_prob)):
            cdf.append(cdf[-1] + class_prob[i])

        self.sequence_idx = [bisect(cdf, random()) for i in range(self.dataset_size)]

        # classes
        self.sequence_class = self.class_labels[self.sequence_idx]

        # frequencies
        self.frequencies = [uniform(freq_range0, freq_range1) for _ in range(self.dataset_size)]

        # amplitudes
        self.amplitudes = [uniform(0, 1) for _ in range(self.dataset_size)]

        # phases
        self.phases = [uniform(0, 2 * numpy.pi) for _ in range(self.dataset_size)]

    def create_dataset_in_disk(self):
        pass

    def get_num_batches(self):
        return int(numpy.ceil(self.dataset_size/self.batch_size))

    def get_item_shape(self):
        return self.batch_size, int(self.duration * self.fs), 1

    def get_one_hot_encoding(self, label):
        roll = numpy.zeros((1, len(self.class_labels)))
        pos = numpy.where(self.class_labels == label)
        roll[:, pos] = 1
        return roll

    def process_output(self, batch_data, batch_labels):
        x_training = numpy.array(batch_data)
        y_training = numpy.vstack([x for x in batch_labels])

        return x_training, y_training

    def flow(self):
        # for keras.model.fit_generator()

        while True:
            batch_idx = 0
            for idx in range(self.dataset_size):

                if batch_idx == 0:
                    batch_data = []
                    batch_labels = []

                curr_label = self.sequence_class[idx]
                curr_freq = self.frequencies[idx]
                curr_amp = self.amplitudes[idx]
                curr_phase = self.phases[idx]

                if curr_label != "whiteNoise":
                    curr_signal = getattr(self, curr_label)(curr_freq, curr_phase)
                    curr_signal = curr_signal.reshape(-1, 1)
                    curr_signal *= curr_amp
                    batch_data.append(curr_signal)
                else:
                    curr_signal = getattr(self, curr_label)()
                    curr_signal = curr_signal.reshape(-1, 1)
                    curr_signal *= curr_amp
                    batch_data.append(curr_signal)

                batch_labels.append(self.get_one_hot_encoding(curr_label))

                if batch_idx == self.batch_size - 1:
                    yield self.process_output(batch_data, batch_labels)
                else:
                    batch_idx += 1

            if not batch_idx == 0:
                yield self.process_output(batch_data, batch_labels)

    def get_signal_types(self):
        return self.signal_types

    def sine(self, f, phi):
        return numpy.sin(numpy.arange(self.n) * numpy.pi * 2 * f / float(self.n) + phi)

    def sawtooth(self, f, phi):
        return numpy.mod(numpy.arange(self.n) * 2.0 * f / self.n + f / 2.0 + phi, 2) - 1

    def square(self, f, phi):
        return (self.sawtooth(f, phi) >= 0) * 2.0 - 1

    def tri(self, f, phi):
        osc = 2 * self.sawtooth(f, phi) * self.square(f, phi) - 1
        # This needs a phase shift on top:
        sh = int(self.n / f / 4.0)
        return numpy.concatenate((osc[sh:], osc[:sh]))

    def whiteNoise(self):
        return numpy.random.random(self.n) * 2.0 - 1

    def sineClip(self, f, phi):
        """
            sineClip: 'r' sets the point at which to clip the sinewave, in [0,1]; e.g.
            r = 0.8 clips the sinewave at 80% intensity, both top and bottom.
        """
        return numpy.clip(self.sine(f, phi), -self.clip_r, self.clip_r)

    def sineTrunc(self, f, phi):
        """
            Truncated sinewave; t0 and t1 set start and end time for the
            sinewave, as ratios relative to N (0 < t0 < t1 < 1). For instance, t0=0.1
            and t1 = 0.9 means to cut off the sinewave at 10% of the start and end.
        """
        r = numpy.arange(self.n) / float(self.n)
        envelope = (r > self.trunc_t0) * (r <= self.trunc_t1) * 1
        return self.sine(f, phi) * envelope