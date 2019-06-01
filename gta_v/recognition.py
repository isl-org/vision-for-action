import numpy as np
import scipy.ndimage.filters as filters

import image_utils.preprocess as image_preprocess

DEBUG = False
MAPPING = {
        'final': (0, 4),
        'disparity': (4, 5),
        'object_id': (5, 6),
        'texture_id': (6, 7),
        'flow': (7, 9),
        'albedo': (9, 13),
        'velocity': (13, 15),
        'final_prev': (15, 19),
        }
PREPROCESS_MAP = {
        # Image.
        'final': image_preprocess.normalize_func(255.0, 0.5),

        # Depth. 1 + 1 + 3
        'disparity': image_preprocess.clip_and_scale(0.1),
        'depth': image_preprocess.clip_and_scale(100.0),
        'normal': image_preprocess.normalize_func(2.0, 0.0),

        # Label. 4 + 1 + 1
        'semantic': image_preprocess.identity_func,
        'boundary': image_preprocess.identity_func,

        # Flow. 2 + 2 + 2
        'flow': image_preprocess.normalize_flow(32.0),
        'dynamic_flow': image_preprocess.normalize_flow(32.0),
        'static_flow': image_preprocess.normalize_flow(32.0),

        # Material. 4
        'albedo': image_preprocess.normalize_func(255.0, 0.5),

        # Past. 4
        'final_prev': image_preprocess.normalize_func(255.0, 0.5),
        }


def debug(result_raw):
    import cv2
    import image_utils.visualize as vis

    for name, data in result_raw:
        print('%s: %.3f %.3f %.3f %d' % (
            name,
            data.min(), data.mean(), data.max(),
            np.isnan(data).sum()))

        if name in ['final', 'albedo', 'final_prev']:
            cv2.imshow(name, cv2.cvtColor(np.uint8(data[:,:,:3]), cv2.COLOR_BGR2RGB))
        if name in ['disparity', 'depth']:
            cv2.imshow(name, (data - data.min()) / (data.max() - data.min()))
        if name == 'boundary':
            cv2.imshow(name, np.uint8(data * 255))
        if 'flow' in name:
            cv2.imshow(name, vis.vis_flow(data, 10, 14))
        if name == 'normal':
            cv2.imshow(name, vis.vis_normals(data))
        if name == 'semantic':
            for i in range(data.shape[2]):
                cv2.imshow('%s_%s' % (name, i), np.uint8(data[:,:,i] * 255))

    cv2.waitKey(100)


def take_input(name, inputs):
    i, j = MAPPING[name]

    return inputs[:,:,i:j]


def maybe_preprocess(inputs, name, should_preprocess):
    if should_preprocess:
        return PREPROCESS_MAP[name](inputs)

    return inputs


def get_normals(depth, mult=1.0):
    padded_depth = np.pad(np.squeeze(depth), 1, mode='edge')

    u = (padded_depth[1:-1,2:] - padded_depth[1:-1,:-2]) / 2.0
    v = (padded_depth[2:,1:-1] - padded_depth[:-2,1:-1]) / 2.0

    normals = np.stack([-u, -v, -np.ones((depth.shape[0], depth.shape[1]), dtype=np.float32) * mult], -1)
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,np.newaxis]

    return normals


def get_class_id(object_id):
    return ((object_id >> 28) & 0xf)


def get_class_id_one_hot(class_id, n_classes):
    one_hot = np.zeros((class_id.shape[0], class_id.shape[1], 2), dtype=np.uint8)

    one_hot[class_id[:,:,0] == 1,0] = 1
    one_hot[class_id[:,:,0] == 2,1] = 1

    return one_hot


def get_rgb_from_one_hot(one_hot):
    import constants

    tmp = np.zeros((one_hot.shape[0], one_hot.shape[1], 3), dtype=np.uint8)

    for i in range(one_hot.shape[-1]):
        tmp[one_hot[:,:,i] == 1] += np.uint8(constants.colors_car[i])

    return tmp


def get_boundary(id_buffer):
    mins = filters.minimum_filter(id_buffer, size=(2,2,1))
    maxs = filters.maximum_filter(id_buffer, size=(2,2,1))

    return np.uint8(mins != maxs)


def get_boundary_new(id_buffer, semantics_one_hot):
    tmp = id_buffer
    primes = [103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
            167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]

    # HACK.
    for i in range(semantics_one_hot.shape[-1]):
        tmp += primes[i] * semantics_one_hot[:,:,i:i+1]

    mins = filters.minimum_filter(tmp, size=(2,2,1))
    maxs = filters.maximum_filter(tmp, size=(2,2,1))

    return np.uint8(mins != maxs)


def get_semantics(inputs):
    object_id = np.uint32(take_input('object_id', inputs))

    # HACK: assume n_classes = 4.
    class_id_one_hot = get_class_id_one_hot(get_class_id(object_id), 4)

    return class_id_one_hot


def get_final(inputs):
    return take_input('final', inputs)


def get_extra_labels(texture_id, labels):
    # Hack.
    mask = np.zeros((texture_id.shape[0], texture_id.shape[1], len(labels)), dtype=np.uint8)

    for i in range(len(labels)-1):
        for x in labels[i]:
            mask[texture_id[:,:,0] == x,i] = 1

    return mask


def get_full_semantics(semantics, extra_labels):
    one_hot = np.concatenate([semantics, extra_labels], -1)
    one_hot[one_hot.sum(axis=2) == 0,-1] = 1

    return one_hot


def get_texture_id(inputs):
    texture_id = np.uint32(take_input('texture_id', inputs))
    object_id = np.uint32(take_input('object_id', inputs))
    class_id = get_class_id(object_id)

    # HACK.
    texture_id[class_id == 3] = object_id[class_id == 3]
    texture_id[texture_id == 0] = object_id[texture_id == 0]

    return texture_id


def get_inputs(desired, inputs, road_ids, should_preprocess):
    """
    desired should be a subset of ['image', 'depth', 'label', 'flow', 'material'].
    """
    result_raw = list()

    if 'image' in desired:
        final = take_input('final', inputs)

        result_raw.append(('final', final))

    if 'depth' in desired:
        disparity = take_input('disparity', inputs) + 1e-6

        depth = 1.0 / disparity
        normal = get_normals(depth)

        result_raw.append(('disparity', disparity))
        result_raw.append(('depth', depth))
        result_raw.append(('normal', normal))

    if 'label' in desired:
        texture_id = np.uint32(take_input('texture_id', inputs))
        object_id = np.uint32(take_input('object_id', inputs))
        class_id = get_class_id(object_id)
        class_id_one_hot = get_class_id_one_hot(class_id, 4)

        extra_labels = get_extra_labels(texture_id, road_ids)
        semantic = get_full_semantics(class_id_one_hot, extra_labels)

        boundary = get_boundary_new(object_id, extra_labels)

        result_raw.append(('semantic', semantic))
        result_raw.append(('boundary', boundary))

    if 'flow' in desired:
        flow = take_input('flow', inputs)
        dynamic_flow = take_input('velocity', inputs)
        static_flow = flow - dynamic_flow

        result_raw.append(('flow', flow))
        result_raw.append(('dynamic_flow', dynamic_flow))
        result_raw.append(('static_flow', static_flow))

    if 'material' in desired:
        albedo = take_input('albedo', inputs)

        result_raw.append(('albedo', albedo))

    if 'past' in desired:
        final_prev = take_input('final_prev', inputs)

        result_raw.append(('final_prev', final_prev))

    if not should_preprocess:
        if DEBUG:
            debug(result_raw)

        return np.concatenate([x[1] for x in result_raw], -1)

    result = list()

    for name, raw_buffer in result_raw:
        result.append(PREPROCESS_MAP[name](raw_buffer))

    return np.concatenate(result, -1)
