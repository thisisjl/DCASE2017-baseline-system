import copy
import numpy
from .files import AudioFile
from .features import FeatureContainer


class RawAudioBatcher():
    def __init__(self, split_files, _annotations, class_labels, batch_size=1, mono=True,
                 desired_fs=22050, segment=True, frame_size_sec0=5.0):
        self.files = split_files
        self.annotations = _annotations
        self.batch_size = batch_size
        self.mono = mono
        self.desired_fs = desired_fs
        self.segment = segment
        self.frame_size_smp0 = int(frame_size_sec0 * desired_fs)

        self.generator_sequence = copy.copy(self.files)


        self.n_frames = None
        self.frame_size_smp = None
        self.n_channels = None
        self.duration_smp = None
        self.class_labels = class_labels

        self.item_shape = self.get_item_shape()

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
        n_channels = numpy.shape(audio)[1]
        start = 0
        end = self.frame_size_smp

        frame_matrix = numpy.zeros((self.n_frames, self.frame_size_smp, n_channels))

        for f_idx in range(self.n_frames):
            frame_matrix[f_idx, :, :] = audio[start:end, :]

            start += self.frame_size_smp
            end = start + self.frame_size_smp

        return frame_matrix

    def generator(self):
        # split_files = self.files
        _annotations = self.annotations

        while True:

            batch_idx = 0

            # for item, metadata in _annotations.items():
            for item_filename in self.generator_sequence:

                if batch_idx == 0:
                    batch_files = []
                    batch_data = {}
                    batch_annotations = {}

                batch_files.append(item_filename)
                batch_annotations[item_filename] = _annotations[item_filename]  # metadata

                af = AudioFile(filename=item_filename)
                item_data_len = int(numpy.ceil(af.info.duration * self.desired_fs))

                item_data0, fs = AudioFile().load(item_filename, fs=self.desired_fs, mono=self.mono)

                item_data = item_data0.reshape((item_data_len, self.n_channels))
                item_data = item_data[:int(numpy.ceil(self.duration_smp / fs - 1 / fs) * fs), :]

                if self.segment:
                    # TODO: segment with hop_size
                    item_data = self.create_segments(item_data)

                fc = FeatureContainer()
                fc.feat = [item_data]  # [item_data.reshape(1, -1)]

                batch_data[item_filename] = fc

                if batch_idx == self.batch_size - 1:

                    # Convert annotations into activity matrix format
                    activity_matrix_dict = self._get_target_matrix_dict(data=batch_data,
                                                                        annotations=batch_annotations)

                    x_training = numpy.vstack([batch_data[x].feat[0] for x in batch_files])
                    y_training = numpy.vstack([activity_matrix_dict[x] for x in batch_files])

                    self.generator_sequence = self.generator_sequence[batch_idx+1:]

                    batch_idx = 0  # reinitialize batch counter
                    yield x_training, y_training  # output of generator

                else:
                    batch_idx += 1

            if not batch_idx == 0:
                activity_matrix_dict = self._get_target_matrix_dict(data=batch_data,
                                                                    annotations=batch_annotations)
                x_training = numpy.vstack([batch_data[x].feat[0] for x in batch_files])
                y_training = numpy.vstack([activity_matrix_dict[x] for x in batch_files])
                yield x_training, y_training  # output of generator

    def _get_target_matrix_dict(self, data, annotations):
        activity_matrix_dict = {}
        for audio_filename in sorted(list(annotations.keys())):
            frame_count = data[audio_filename].feat[0].shape[0]
            pos = self.class_labels.index(annotations[audio_filename]['scene_label'])
            roll = numpy.zeros((frame_count, len(self.class_labels)))
            roll[:, pos] = 1
            activity_matrix_dict[audio_filename] = roll
        return activity_matrix_dict
