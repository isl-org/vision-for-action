import collections

import numpy as np
import tensorflow as tf

from vizdoom_utils import label
from vizdoom_utils import flow
from vizdoom_utils import normal


RGB = 'RGB'
GRAY = 'GRAY'

INPUTS = [
        'image', 'depth', 'label', 'flow', 'normal',
        ]


SensorArguments = collections.namedtuple('SensorArguments',
        ['color_mode',
            'enable_image', 'enable_depth', 'enable_label',
            'enable_flow', 'enable_normal'])


def identity_func(x):
    return x


def normalize_func(scale, offset):
    def func(x):
        return x / scale - offset

    return func


def normalize_flow(max_displacement):
    def func(x):
        x = np.clip(x, -max_displacement, max_displacement)
        x = x / (2.0 * max_displacement)

        return x

    return func


# Has to be after the function declarations.
PREPROCESS_MAP = {
        'image': normalize_func(255.0, 0.5),
        'depth': normalize_func(255.0, 0.0),
        'label': identity_func,
        'flow': normalize_flow(32.0),
        'normal': normalize_func(32.0, 0.0),
        }


def get_color_mode(string_form):
    if string_form.upper() in [RGB, GRAY]:
        return string_form

    raise ValueError('%s not valid color mode.' % string_form)


# Assumes (n, h, w, c).
def rip_out_image(inputs, args):
    if args.color_mode == RGB:
        return inputs[:,:,:,:3]

    return inputs[:,:,:,:1]


def rip_out_others(inputs, args):
    if args.color_mode == RGB:
        return inputs[:,:,:,3:]

    return inputs[:,:,:,1:]


def get_loss_tf(pred, target, args):
    def _l1(x, y):
        if len(x.shape) == 3:
            axis = [1, 2]
        else:
            axis = [1, 2, 3]

        return tf.reduce_mean(tf.reduce_sum(tf.abs(x - y), axis))

    mapping = get_channel_mapping(args, include_image=False)

    loss = 0.0

    if args.enable_depth:
        i = mapping['depth'][0]
        j = mapping['depth'][-1] + 1

        loss += _l1(pred[:,:,:,i:j+1], target[:,:,:,i:j+1])

    if args.enable_label:
        i = mapping['label'][0]
        j = mapping['label'][-1] + 1

        loss += tf.losses.softmax_cross_entropy(
                target[:,:,:,i:j+1], pred[:,:,:,i:j+1])

    if args.enable_flow:
        i = mapping['flow'][0]
        j = mapping['flow'][-1] + 1

        loss += _l1(pred[:,:,:,i:j+1], target[:,:,:,i:j+1])

    if args.enable_normal:
        i = mapping['normal'][0]
        j = mapping['normal'][-1] + 1

        loss += _l1(pred[:,:,:,i:j+1], target[:,:,:,i:j+1])

    return loss


def get_channel_mapping(args, include_image=True):
    result = dict()

    idx = 0
    idx_new = 0

    if include_image and args.enable_image:
        idx_new += int(args.color_mode == RGB) * 3
        idx_new += int(args.color_mode == GRAY)
        result['image'] = tuple(range(idx, idx_new))
        idx = idx_new

    if args.enable_depth:
        idx_new += 1
        result['depth'] = tuple(range(idx, idx_new))
        idx = idx_new

    if args.enable_label:
        idx_new += label.N_SEMANTICS
        result['label'] = tuple(range(idx, idx_new))
        idx = idx_new

    if args.enable_flow:
        idx_new += 2
        result['flow'] = tuple(range(idx, idx_new))
        idx = idx_new

    if args.enable_normal:
        idx_new += 3
        result['normal'] = tuple(range(idx, idx_new))
        idx = idx_new

    return result


def get_num_channels(args):
    channels = 0
    channels += int(args.enable_image) * int(args.color_mode == RGB) * 3
    channels += int(args.enable_image) * int(args.color_mode == GRAY)
    channels += int(args.enable_depth)
    channels += args.enable_label * label.N_SEMANTICS
    channels += args.enable_flow * 2
    channels += args.enable_normal * 3

    return channels


def get_num_channels_image(args):
    channels = 0
    channels += int(args.enable_image) * int(args.color_mode == RGB) * 3
    channels += int(args.enable_image) * int(args.color_mode == GRAY)

    return channels


def get_num_channels_other(args):
    return get_num_channels(args) - get_num_channels_image(args)


class VizdoomSensors(object):
    def __init__(self, game, color_mode,
            enable_image=False, enable_depth=False,
            enable_label=False, enable_flow=False, enable_normal=False):
        self.game = game

        self.color_mode = color_mode

        self.enable_image = enable_image
        self.enable_depth = enable_depth
        self.enable_label = enable_label
        self.enable_flow = enable_flow
        self.enable_normal = enable_normal

        self.height = self.game.get_screen_height()
        self.width = self.game.get_screen_width()

        self.reset()

        self.channels = get_num_channels(self)

        if self.enable_depth or self.enable_normal:
            self.game.set_depth_buffer_enabled(True)

        if self.enable_label:
            self.game.set_labels_buffer_enabled(True)

        # Init flow.
        self.flow_computer = flow.FlowComputer(self.width, self.height)

        flow.set_required_game_variables(game)

    def init_from_args(game, sensor_args):
        if sensor_args.color_mode == RGB:
            raise Exception('Only gray supported.')

        return VizdoomSensors(
                game, sensor_args.color_mode,
                sensor_args.enable_image, sensor_args.enable_depth,
                sensor_args.enable_label, sensor_args.enable_flow, sensor_args.enable_normal)

    def reset(self):
        """
        Store last time steps in these buffers.
        """
        self.image_buffer = None
        self.depth_buffer = None
        self.label_buffer = None
        self.flow_buffer = None
        self.normal_buffer = None

    def tick(self):
        state = self.game.get_state()

        if not state:
            return

        if self.enable_flow:
            self.flow_computer.tick(self.game)
            self.flow_computer.compute_flow()

        if self.enable_image:
            self.image_buffer = state.screen_buffer

        if self.image_buffer is not None and self.color_mode == RGB:
            self.image_buffer = self.image_buffer.transpose((1, 2, 0))

        if self.enable_depth:
            self.depth_buffer = state.depth_buffer

        if self.enable_label:
            self.label_buffer = label.transform_labels(state.labels, state.labels_buffer)

        if self.enable_flow:
            self.flow_buffer = self.flow_computer.get_flow()

        if self.enable_normal:
            self.normal_buffer = normal.get_normals(state.depth_buffer)

    def get_all(self, preprocess):
        return stack_inputs({
                'image': self.image_buffer,
                'depth': self.depth_buffer,
                'label': self.label_buffer,
                'flow': self.flow_buffer,
                'normal': self.normal_buffer,
                }, preprocess)


def stack_inputs(inputs_dict, preprocess):
    """
    Returns input of shape (C, H, W).

    NOTE(bradyz): bad stuff happens with RGB.
    """
    def get_and_maybe_preprocess(name):
        input_buffer = inputs_dict.get(name)

        if input_buffer is None:
            return None
        elif preprocess:
            return PREPROCESS_MAP[name](input_buffer)

        return input_buffer

    inputs = [get_and_maybe_preprocess(x) for x in INPUTS]
    inputs = [x for x in inputs if x is not None]
    inputs = [x if x.ndim == 3 else np.expand_dims(x, -1) for x in inputs]
    inputs = np.float32(np.concatenate(inputs, -1))
    inputs = inputs.transpose(2, 0, 1)
    inputs[np.isnan(inputs)] = 0.0

    return inputs
