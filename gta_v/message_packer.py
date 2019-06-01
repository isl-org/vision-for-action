import logging
import numpy as np


HEIGHT = 600
WIDTH = 800

DTYPE_TO_SIZE = {
        np.uint8: 1,
        np.uint32: 4,
        np.float32: 4,
        np.double: 8,
        }

INPUTS_INFO = [
        ('final', (np.uint8, (HEIGHT, WIDTH, 4))),
        ('disparity', (np.float32, (HEIGHT, WIDTH, 1))),
        ('object_id', (np.uint32, (HEIGHT, WIDTH, 1))),
        ('texture_id', (np.uint32, (HEIGHT, WIDTH, 1))),
        ('flow', (np.float32, (HEIGHT, WIDTH, 2))),
        ('albedo', (np.uint8, (HEIGHT, WIDTH, 4))),
        ('velocity', (np.float32, (HEIGHT, WIDTH, 2))),
        ]
SERVER_INFO = INPUTS_INFO + [
        ('server_key', (np.uint32, (1,))),
        ('timestamp', (np.double, (1,))),

        ('env_status', (np.uint8, (1,))),
        ('agent_status', (np.uint8, (1,))),

        ('actions_server', (np.float32, (16,))),
        ('reward', (np.float32, (1,))),
        ('reward_curve', (np.float32, (1,))),
        ('position', (np.float32, (3,))),
        ]
CLIENT_INFO = [
        ('client_key', (np.uint32, (1,))),
        ('timestamp', (np.double, (1,))),

        ('env_type', (np.uint8, (1,))),
        ('agent_type', (np.uint8, (1,))),

        ('request_restart', (np.uint8, (1,))),
        ('sync', (np.uint8, (1,))),

        ('actions_client', (np.float32, (16,))),
        ('scenario_params_int', (np.uint8, (32,))),
        ('scenario_params_float', (np.float32, (32,))),
        ]


def maybe_init_mmap(filename, dtype, mode, size):
    try:
        return np.memmap(filename, dtype=dtype, mode=mode, shape=(size,))
    except Exception as e:
        logging.warn(filename)
        logging.warn(e)
        logging.warn('Size: %d.' % size)

    return None


def _product(shape):
    result = 1

    for x in shape:
        result *= x

    return result


def _get_size(data_info):
    return DTYPE_TO_SIZE[data_info[0]] * _product(data_info[1])


def get_info_size(info):
    return sum([_get_size(data_info) for _, data_info in info])


def encode(output_buffer, data_dict, info):
    i = 0

    for data_key, data_info in info:
        dtype, shape = data_info
        size = _get_size(data_info)

        if data_key in data_dict:
            data = data_dict[data_key]

            try:
                if not isinstance(data,np.ndarray) or data_key == 'server_key':
                    data = np.frombuffer(np.squeeze(dtype(data)).tobytes(), dtype=np.uint8)

                real_size = min(size, data.size)

                output_buffer[i:i+real_size] = data
            except Exception as e:
                logging.warn(e)
                logging.warn('Encoding %s failed.' % data_key)
                logging.warn('%s' % data_dict[data_key].shape)

        i += size


def decode(input_buffer, info):
    """
    Expects input_buffer to be a uint8 array (can be mmap).
    """
    byte_buffer = input_buffer.tobytes()

    result = dict()
    i = 0

    for data_key, data_info in info:
        dtype, shape = data_info
        size = _get_size(data_info)

        result[data_key] = np.frombuffer(byte_buffer[i:i+size], dtype=dtype).reshape(shape).squeeze()

        i += size

    return result
