#!/usr/bin/env python3
import cv2
import numpy as np
import vizdoom

from skimage import color


# Debugging.
VISUALIZE_BUFFERS = 0
VISUALIZE_FLOW = 1
VISUALIZE_FLOW_COLORS = 0

# Arrow colors.
FLOW_COLORS = [tuple(255 * int(bit) for bit in bin(i)[2:].rjust(3, '0')) for i in range(8)]
SLEEPTIME_MS = 0.01


def _angle_to_rad(angle):
    return angle * (np.pi / 180.0)


def _rotate_x(roll):
    theta = _angle_to_rad(roll)

    c = np.cos(theta)
    s = np.sin(theta)

    return np.float32([
        [ 1,  0,  0],
        [ 0,  c,  s],
        [ 0, -s,  c]])


def _rotate_y(pitch):
    theta = _angle_to_rad(pitch)

    c = np.cos(theta)
    s = np.sin(theta)

    return np.float32([
        [ c,  0, -s],
        [ 0,  1,  0],
        [ s,  0,  c]])


def _rotate_z(yaw):
    theta = _angle_to_rad(yaw)

    c = np.cos(theta)
    s = np.sin(theta)

    return np.float32([
        [ c,  s,  0],
        [-s,  c,  0],
        [ 0,  0,  1]])


# Doom uses
#   x
#   |
#    -- z
#  /
# y
def _make_rotation(angle, pitch, roll, is_affine=True):
    I = np.eye(4)

    X = _rotate_x(-pitch)
    Y = _rotate_y(-angle)
    Z = _rotate_z(roll)

    result = X.dot(Y.dot(Z))

    if is_affine:
        return _make_affine(result)

    return result


def _make_translation(x, y, z):
    return np.float32([
        [1, 0, 0,  y],
        [0, 1, 0,  z],
        [0, 0, 1, -x],
        [0, 0, 0,  1]])


# http://www.songho.ca/opengl/gl_projectionmatrix.html
def _make_projection(far, near, width, height):
    # Look down negative z axis, so need to negate.
    f = -float(far)
    n = -float(near)
    w = float(width)
    h = float(height)

    return np.float32([
        [n / width, 0, 0, 0],
        [0, n / height, 0, 0],
        [0, 0, -(f + n) / (f - n), -(f * n) / (f - n)],
        [0, 0, -1, 0]])


def _make_affine(matrix_3x3):
    result = np.zeros((4, 4), dtype=np.float32)

    result[:3,:3] = matrix_3x3
    result[3,3] = 1

    return result


def set_required_game_variables(game):
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    for camera_var in Camera.GAME_VARS:
        game.add_available_game_variable(camera_var)


def visualize_flow_colors(flow):
    height, width, _ = flow.shape
    norms = np.sqrt(flow[:,:,0] ** 2 + flow[:,:,1] ** 2)

    hsv = np.ones((height, width, 3), dtype=np.float32)
    hsv[:,:,0] = (np.arctan2(flow[:,:,1], flow[:,:,0]) / np.pi + 1.0) / 2.0
    hsv[:,:,1] = np.minimum(1.0, norms / 100.0)

    rgb = color.hsv2rgb(hsv)
    rgb[np.any(np.isnan(flow), axis=2)] = 0.0

    return rgb


def debug_flow(cur_final, flow_buffer, skip=4, name='Flow Arrows', eps=1e-5):
    cur_final = cur_final.copy()

    color_idx = 0

    for y in range(0, flow_buffer.shape[0], skip):
        for x in range(0, flow_buffer.shape[1], skip):
            dx, dy = flow_buffer[y,x]

            xx = int(round(x + dx))
            yy = int(round(y + dy))

            cv2.arrowedLine(cur_final, (x, y), (xx, yy), FLOW_COLORS[color_idx], 1)

            color_idx = (color_idx + 1) % len(FLOW_COLORS)

    cv2.imshow(name, cur_final)


class Camera(object):
    NEAR = 200
    FAR = 1000

    GAME_VARS = [
            vizdoom.GameVariable.POSITION_X,
            vizdoom.GameVariable.POSITION_Y,
            vizdoom.GameVariable.POSITION_Z,
            vizdoom.GameVariable.ANGLE,
            vizdoom.GameVariable.PITCH,
            vizdoom.GameVariable.ROLL,
            ]

    def __init__(self, x=None, y=None, z=None,
            angle=None, pitch=None, roll=None,
            width=None, height=None):
        self.x = x
        self.y = y
        self.z = z

        self.angle = angle
        self.pitch = pitch
        self.roll = roll

        self.width = width
        self.height = height

        if self.is_valid():
            self.rotation = _make_rotation(self.angle, self.pitch, self.roll)
            self.translation = _make_translation(self.x, self.y, self.z)

            self.view_matrix = self.rotation.dot(self.translation)
            self.proj_matrix = _make_projection(
                    Camera.FAR, Camera.NEAR,
                    self.width, self.height)
        else:
            self.rotation = None
            self.translation = None
            self.view_matrix = None
            self.proj_matrix = None

    def is_valid(self):
        for parameter in [self.x, self.y, self.z, self.angle, self.pitch, self.roll]:
            if parameter is None:
                return False
        return True

    def init(game):
        x, y, z, angle, pitch, roll = map(game.get_game_variable, Camera.GAME_VARS)
        width, height = game.get_screen_width(), game.get_screen_height()

        return Camera(x, y, z, angle, pitch, roll, width, height)

    def copy(self):
        return Camera(
                self.x, self.y, self.z,
                self.angle, self.pitch, self.roll,
                self.width, self.height)

    def __str__(self):
        result = list()

        result.append('x: %d' % self.x)
        result.append('y: %d' % self.y)
        result.append('z: %d' % self.z)
        result.append('angle: %d' % self.angle)
        result.append('pitch: %d' % self.pitch)
        result.append('roll: %d' % self.roll)

        return '\t'.join(result)


class Buffers(object):
    def __init__(self, screen=None, depth=None, labels=None):
        self.screen = screen
        self.depth = depth
        self.labels = labels

    def copy(self):
        return Buffers(self.screen, self.depth, self.labels)

    def is_valid(self):
        return self.screen is not None and self.depth is not None and self.labels is not None


class Sprite(object):
    def __init__(self, uuid, name, x, y, height, width):
        self.uuid = uuid
        self.name = name
        self.x = x
        self.y = y
        self.height = height
        self.width = width

    def init(label):
        return Sprite(
                label.object_id, label.object_name, label.x, label.y,
                label.height, label.width)


class FlowComputer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.prev_camera = None
        self.prev_buffers = None
        self.prev_objects = None

        self.cur_camera = None
        self.cur_buffers = None
        self.cur_objects = None

        self.x_pixel, self.y_pixel = np.meshgrid(range(self.width), range(self.height))
        self.tmp = np.zeros((self.height, self.width, 4))

        self.static_flow = np.zeros((self.height, self.width, 2))
        self.object_flow = np.zeros((self.height, self.width, 2))

        self.clear()

    def clear(self):
        self.prev_camera = Camera()
        self.prev_buffers = Buffers()
        self.prev_objects = dict()

        self.cur_camera = Camera()
        self.cur_buffers = Buffers()
        self.cur_objects = dict()

    def tick(self, game):
        state = game.get_state()

        # Invalid state.
        if state is None:
            self.clear()
            return

        self.prev_objects = self.cur_objects.copy()
        self.cur_objects = {label.object_id: Sprite.init(label) for label in state.labels}

        self.prev_camera = self.cur_camera.copy()
        self.cur_camera = Camera.init(game)

        self.prev_buffers.screen = self.cur_buffers.screen
        self.cur_buffers.screen = state.screen_buffer

        self.prev_buffers.labels = self.cur_buffers.labels
        self.cur_buffers.labels = state.labels_buffer

        self.prev_buffers.depth = self.cur_buffers.depth

        if state.depth_buffer is not None:
            self.cur_buffers.depth = state.depth_buffer * (100.0 / 14.0) + 1e-8
        else:
            self.cur_buffers.depth = None

        if VISUALIZE_BUFFERS:
            cv2.imshow('Depth', state.depth_buffer)
            cv2.waitKey(1)

    def compute_flow(self):
        """
        Must be called after tick and before get_flow.
        """
        if self.has_enough_info():
            self._compute_static_flow()
            self._compute_dynamic_flow()

    def _compute_static_flow(self):
        VV = self.prev_camera.view_matrix
        PP = self.prev_camera.proj_matrix

        V = self.cur_camera.view_matrix
        P = self.cur_camera.proj_matrix

        V_inv = np.linalg.inv(V)
        P_inv = np.linalg.inv(P)

        reproject_matrix = PP.dot(VV.dot(V_inv.dot(P_inv)))

        # Hard to read code for performance purposes. Each line was benchmarked.
        self.tmp[:] = -self.cur_buffers.depth[:,:,np.newaxis]
        self.tmp[:,:,0] *= 2.0 * (self.x_pixel / self.width - 0.5)
        self.tmp[:,:,1] *= 2.0 * (self.y_pixel / self.height - 0.5)
        self.tmp[:,:,2] *= -P[2,2]
        self.tmp[:,:,2] += P[2,3]
        self.tmp = self.tmp.dot(reproject_matrix.T)
        self.tmp[:,:,3] += 1e-8

        self.static_flow[:,:,0] = (self.tmp[:,:,0] / self.tmp[:,:,3] / 2.0 + 0.5) * self.width - self.x_pixel
        self.static_flow[:,:,1] = (self.tmp[:,:,1] / self.tmp[:,:,3] / 2.0 + 0.5) * self.height - self.y_pixel

    def _compute_dynamic_flow(self):
        self.object_flow[:] = 0.0

        for uuid, sprite in self.cur_objects.items():
            if uuid not in self.prev_objects:
                continue

            sprite_prev = self.prev_objects[uuid]

            x1, y1 = sprite.x, sprite.y
            x2, y2 = sprite.x + sprite.width, sprite.y + sprite.height

            xx1, yy1 = sprite_prev.x, sprite_prev.y
            xx2, yy2 = sprite_prev.x + sprite_prev.width, sprite_prev.y + sprite_prev.height

            q11 = (xx1 - x1, yy1 - y1)
            q12 = (xx1 - x1, yy2 - y2)
            q21 = (xx2 - x2, yy1 - y1)
            q22 = (xx2 - x2, yy2 - y2)

            # Hard to read code for performance purposes.
            x, y = np.meshgrid(
                    range(sprite.x, sprite.x + sprite.width+1),
                    range(sprite.y, sprite.y + sprite.height+1))

            self.object_flow[y1:y2+1,x1:x2+1,0] = (1.0 / ((x2 - x1) * (y2 - y1) + 1e-7)) * (
                    q11[0] * (x2 - x) * (y2 - y) + q21[0] * (x - x1) * (y2 - y) +
                    q12[0] * (x2 - x) * (y - y1) + q22[0] * (x - x1) * (y - y1))
            self.object_flow[y1:y2+1,x1:x2+1,1] = (1.0 / ((x2 - x1) * (y2 - y1) + 1e-7)) * (
                    q11[1] * (x2 - x) * (y2 - y) + q21[1] * (x - x1) * (y2 - y) +
                    q12[1] * (x2 - x) * (y - y1) + q22[1] * (x - x1) * (y - y1))

    def has_enough_info(self):
        """
        Has enough information to compute flow.
        """
        if not self.prev_camera.is_valid() or not self.prev_buffers.is_valid():
            return False
        if not self.cur_camera.is_valid() or not self.cur_buffers.is_valid():
            return False

        return True

    def get_zero_flow(self):
        return np.zeros((self.height, self.width, 2))

    def get_object_mask(self):
        return (self.cur_buffers.labels > 1)[:,:,np.newaxis]

    def get_static_flow(self):
        if self.has_enough_info():
            return self.static_flow

        return self.get_zero_flow()

    def get_dynamic_flow(self):
        if self.has_enough_info():
            return self.get_object_mask() * (self.object_flow - self.static_flow)

        return self.get_zero_flow()

    def get_object_flow(self):
        if self.has_enough_info():
            return self.object_flow

        return self.get_zero_flow()

    def get_flow(self):
        if self.has_enough_info():
            object_mask = self.get_object_mask()
            flow = object_mask * self.object_flow + (1 - object_mask) * self.static_flow

            return flow

        return self.get_zero_flow()


if __name__ == '__main__':
    CONFIG = '/home/bzhou/code/VizDoom_clean/scenarios/my_way_home.cfg'
    CONFIG = '/home/bzhou/code/DirectFuturePredictionWithExtraStuff/maps/D2_navigation.cfg'
    CONFIG = '/home/bzhou/code/DirectFuturePredictionWithExtraStuff/maps/D3_battle.cfg'
    CONFIG = '/home/bzhou/code/DirectFuturePredictionWithExtraStuff/maps/D1_basic.cfg'

    game = vizdoom.DoomGame()

    game.load_config(CONFIG)
    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.SPECTATOR)

    flow_computer = FlowComputer(game.get_screen_width(), game.get_screen_height())

    set_required_game_variables(game)

    game.init()

    while True:
        print('New episode.')

        game.new_episode()

        while not game.is_episode_finished():
            flow_computer.tick(game)
            flow_buffer = flow_computer.get_flow()

            game.advance_action(4)

            if VISUALIZE_FLOW:
                debug_flow(flow_computer.cur_buffers.screen, flow_computer.get_dynamic_flow(), name='dynamic')
                debug_flow(flow_computer.cur_buffers.screen, flow_computer.get_object_flow(), name='object')
                debug_flow(flow_computer.cur_buffers.screen, flow_computer.get_static_flow(), name='static')
                cv2.waitKey(1)

            if VISUALIZE_FLOW_COLORS:
                cv2.imshow('Flow Colors', visualize_flow_colors(flow_buffer))
                cv2.waitKey(1)

    cv2.destroyAllWindows()
