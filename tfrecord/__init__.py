import tensorflow as tf

# loosely based on 
# http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html


def pack_int64_list(lst):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=lst)
    )


def unpack_int64_list(feature):
    return feature.int64_list.value


def pack_float_list(lst):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=lst)
    )


def unpack_float_list(feature):
    return feature.float_list.value


def pack_bytes_list(lst):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=lst)
    )


def unpack_bytes_list(feature):
    return feature.bytes_list.value


_tf_compression_revmap = {
    y: x for x, y in tf.python_io.TFRecordOptions.compression_type_map.items()
    if y != ''
}


def compression_code(compression):
    if compression is None:
        return None
    code = _tf_compression_revmap.get(compression)
    if code is None:
        raise ValueError(
            'Unknown or unsupported compression type: ' + compression)


class Writer:
    def __init__(self, fname, pack_sample, compression=None):
        self._engine = tf.python_io.TFRecordWriter(
            fname, compression_code(compression))
        self._pack_sample = pack_sample
        self._closed = False

    def __enter__(self):
        self._engine.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._engine.close()

    def write_sample(self, sample):
        example = tf.train.Example(
            features=tf.train.Features(feature=self._pack_sample(sample))
        )
        return self._engine.write(example.SerializeToString())

    def close(self):
        if not self._closed:
            self._engine.close()
        self._closed = True


class Reader:
    def __init__(self, fname, unpack_sample, compression=None):
        self._engine = iter(tf.python_io.tf_record_iterator(
            fname, compression_code(compression)))
        self._unpack_sample = unpack_sample

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        buffer = next(self._engine)
        example = tf.train.Example()
        example.ParseFromString(buffer)
        return self._unpack_sample(example.features.feature)

    def read_sample(self):
        try:
            return __next__(self)
        except StopIteration:
            return None
