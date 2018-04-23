#  tfrecord
Sample code to write/read TensorFlow records (aka `TFRecord`).

## building

```
virtualenv .venv -p python3.6
. .venv/bin/activate
pip install -r requirements.txt
```

## testing

```
pip install pytest
pytest
```

## notes
TFRecord is a format for storing lists of dictionaries, using Google Protocol Buffers
under the hood. It supports streaming writes and streaming reads, cloud filenames, and
compression. It also does checksumming and adds record boundary guards (not sure if
this is good or not).

Unfortunately, TF API for read/write is low-level, poorly documented (as of TF v1.7), and
counter-intuitive. Specifically:

1. `TFRecordWriter` is a normal class, that behaves as one would expect, but there is no
   symmetrical class `TFRecordReader` (well, there is such a class, but its an Op and
   thus is a very different beast). Instead, TF provides `tf_record_iterator`, as
   a streaming record reader.
2. Reading and writing using this low-level API is tedious as require user
   to build protocol buffer classes (`tf.train.Example`, `tf.train.Features`, etc)

Here is an attempt to provide a bit more convenient API.

## api

Example 1. Each sample is an array of integers

```
FILENAME = 'myfile.tfrecords'

def pack_sample(sample):
    return {
        'vals': pack_int64_list(sample)
    }

def unpack_sample(feats):
    return unpack_int64_list(feats['vals'])

with tfrecord.tfrecord.Writer(FILENAME, pack_sample) as w:
    w.write([1, 2, 3])

with tfrecord.tfrecord.Reader(FILENAME, unpack_sample) as r:
    for sample in r:
        print(sample)
```

Example 2. Storing input and target integer array (e.g. training data for a language model)
```
FILENAME = 'myfile.tfrecords'

def pack_sample(sample):
    return {
        'inputs' : pack_int64_list(sample['inputs']),
        'targets': pack_int64_list(sample['targets'])
    }

def unpack_sample(feats):
    return {
        'inputs' : unpack_int64_list(feats['inputs']),
        'targets': unpack_int64_list(feats['targets'])
    }

with tfrecord.tfrecord.Writer(FILENAME, pack_sample) as w:
    w.write({
        'inputs' : [1, 2, 3],
        'targets': [4, 5, 6]
    })

with tfrecord.tfrecord.Reader(FILENAME, unpack_sample) as r:
    for sample in r:
        print(sample)
```

Example 3. Writing float and bytes arrays
```
FILENAME = 'myfile.tfrecords'

def pack_sample(sample):
    return {
        'floats': pack_float_list(sample['floats']),
        'bytes' : pack_bytes_list(sample['bytes'])
    }

def unpack_sample(feats):
    return {
        'floats': unpack_float_list(feats['floats']),
        'bytes' : unpack_bytes_list(feats['bytes'])
    }

with tfrecord.tfrecord.Writer(FILENAME, pack_sample) as w:
    w.write({
        'floats': [1., 2., 3.],
        'bytes' : [b'hello', b'word']
    })

with tfrecord.tfrecord.Reader(FILENAME, unpack_sample) as r:
    for sample in r:
        print(sample)
```
