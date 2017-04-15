import os
import copy
import numpy
from .files import AudioFile
from .features import FeatureContainer


class RawAudioBatcher():
    def __init__(self, split_files, _annotations, class_labels, batch_size=1, mono=True,
                 desired_fs=22050, segment=True, frame_size_sec0=5.0, normalize=False):
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
        self.annotations = _annotations
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

