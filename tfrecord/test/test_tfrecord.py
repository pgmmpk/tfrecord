import pytest
from tfrecord import (
    Writer, Reader,
    pack_int64_list, unpack_int64_list,
    pack_float_list, unpack_float_list,
    pack_bytes_list, unpack_bytes_list,
)

SAMPLES = [
    {
        'int_sample'  : [1, 2, 3],
        'float_sample': [0.1, 0.2, 0.3],
        'byte_sample' : [b'1', b'2', b'3']
    },
    {
        'int_sample'  : [4, 5],
        'float_sample': [0.4],
        'byte_sample' : [b'hello', b'world!']
    },
]

def pack_sample(sample):
    return {
        'int_sample'  : pack_int64_list(sample['int_sample']),
        'float_sample': pack_float_list(sample['float_sample']),
        'byte_sample' : pack_bytes_list (sample['byte_sample'])
    }

def unpack_sample(feats):
    return {
        'int_sample'  : unpack_int64_list(feats['int_sample']),
        'float_sample': unpack_float_list(feats['float_sample']),
        'byte_sample' : unpack_bytes_list (feats['byte_sample'])
    }

def test_smoke():

    def pack_sample(sample):
        return {
            'vals': pack_int64_list(sample)
        }

    def unpack_sample(feats):
        return unpack_int64_list(feats['vals'])

    with Writer('myfile.tfrecords', pack_sample) as w:
        w.write_sample([1, 2, 3])

    with Reader('myfile.tfrecords', unpack_sample) as r:
        samples = list(r)

    assert samples == [[1, 2, 3]]

@pytest.mark.parametrize('compression', [None, 'ZLIB', 'GZIP'])
def test(compression):
    with Writer('test.tfrecords', pack_sample, compression=compression) as w:
        for sample in SAMPLES:
            w.write_sample(sample)

    with Reader('test.tfrecords', unpack_sample, compression=compression) as r:
        read_samples = list(r)

        # float serialization roundtrip brings minor diffs, ignore
        for s in read_samples:
            s['float_sample'] = [pytest.approx(x) for x in s['float_sample']]

        assert SAMPLES == read_samples
