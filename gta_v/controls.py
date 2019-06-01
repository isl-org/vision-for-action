import logging
import random
import math


import pyhookv as h

import utils


look_left = lambda: h.Controls.set_control_normal(0, h.eControl.control_look_left_right, 1)
look_right = lambda: h.Controls.set_control_normal(0, h.eControl.control_look_left_right, -1)
look_up = lambda: h.Controls.set_control_normal(0, h.eControl.control_look_up_down, 1)
look_down = lambda: h.Controls.set_control_normal(0, h.eControl.control_look_up_down, -1)
move_forward = lambda: h.Controls.set_control_normal(0, h.eControl.control_move_up_down, -1)
fire = lambda: h.Controls.set_control_normal(0, h.eControl.control_attack, 1)
ON_FOOT = {
        0: look_left,
        1: look_right,
        2: look_up,
        3: look_down,
        4: move_forward,
        5: fire,
        }


def act(mapping):
    def func(action):
        if action is None:
            return

        for i in range(len(action)):
            if i in mapping and action[i] == 1:
                mapping[i]()

    return func


def get_player_lookat():
    ped_id = h.Player.ped_id()

    u = h.Vector3(ped_id.forward_vector.x, ped_id.forward_vector.y, 0.0)
    u = utils.normalize(u)

    return u


def get_vector_to(p2):
    p1 = h.Player.ped_id().get_coords(1)

    v = utils.sub(p2, p1)
    v = h.Vector3(v.x, v.y, 0.0)
    v = utils.normalize(v)

    return v


def get_relative_yaw_to_point(p2, offset):
    u = get_player_lookat()
    v = get_vector_to(p2)
    angle = 180 / math.pi * math.asin(u.x * v.y - u.y * v.x)

    return angle - offset


def is_looking_at_point(p2, offset, eps):
    u = get_player_lookat()
    v = get_vector_to(p2)
    angle = 180 / math.pi * math.asin(u.x * v.y - u.y * v.x)

    return abs(angle - offset) < eps

    # logging.info('%s %s' % (abs(1.0 - utils.get_angle(u, v)), eps))
    # logging.info(abs(utils.get_angle(u, v)) < eps)

    return abs(1.0 - utils.get_angle(u, v)) < eps


def is_looking_at_angle(angle, eps=5e-1):
    # logging.info('%s %s' % (h.Cam.get_gameplay_relative_pitch(), angle))
    return abs(h.Cam.get_gameplay_relative_pitch() - angle) < eps


class Controls(object):
    _LAST_ACTION = None
    _LAST_ANGLE = None

    ACTION_VECTOR = list()

    @classmethod
    def has_been_inited(cls):
        return cls._LAST_ACTION is not None

    @classmethod
    def init(cls, action=1.0, angle=0.0):
        cls._LAST_ACTION = action
        cls._LAST_ANGLE = angle

    @classmethod
    def reset(cls):
        """
        Call every frame.
        """
        for i in range(len(cls.ACTION_VECTOR)):
            cls.ACTION_VECTOR[i] = 0.0

    @classmethod
    def get_last_action(cls):
        return cls.ACTION_VECTOR

    @classmethod
    def _record_action(cls, i):
        cls.ACTION_VECTOR[i] = 1.0

    @staticmethod
    def look_up(amount=1.0):
        h.Controls.set_control_normal(0, h.eControl.control_look_up_down,  -amount)

    @staticmethod
    def look_down(amount=1.0):
        h.Controls.set_control_normal(0, h.eControl.control_look_up_down, amount)

    @staticmethod
    def look_left(amount=1.0):
        h.Controls.set_control_normal(0, h.eControl.control_look_left_right, -amount)

    @staticmethod
    def look_right(amount=1.0):
        h.Controls.set_control_normal(0, h.eControl.control_look_left_right, amount)

    @staticmethod
    def move_forward(amount=1.0):
        h.Controls.set_control_normal(0, h.eControl.control_move_up_down, -amount)

    @staticmethod
    def shoot():
        h.Controls.set_control_normal(0, h.eControl.control_attack, 1)

    @classmethod
    def look_left_right(cls, amount):
        if amount < 0:
            cls.look_left(-amount)
        else:
            cls.look_right(amount)

    @classmethod
    def look_at_point(cls, p2, offset, eps=1e-2):
        """
        Aligns the heading.

        NOTE: ignores pitch.
        """
        u = get_player_lookat()
        v = get_vector_to(p2)
        theta = 180 / math.pi * math.asin(u.x * v.y - u.y * v.x)

        if abs(theta - offset) < eps:
            return

        if theta > offset:
            cls.look_left()
        else:
            cls.look_right()

    @classmethod
    def look_at_angle(cls, angle, eps=5e0):
        """
        angle is [-1, 1].
        """
        if is_looking_at_angle(angle, eps):
            return

        if h.Cam.get_gameplay_relative_pitch() < angle:
            cls.look_up(1.0)
        else:
            cls.look_down(1.0)


class BasicControls(Controls):
    LOOK_UP = 0
    LOOK_DOWN = 1
    LOOK_LEFT = 2
    LOOK_RIGHT = 3
    MOVE_FORWARD = 4
    SHOOT = 5

    # Hack for deer.
    ACTION_VECTOR = [0.0 for _ in range(6)] + [0.0 for _ in range(2)]

    @classmethod
    def look_up(cls, amount=1.0):
        super().look_up()
        cls._record_action(cls.LOOK_UP)

    @classmethod
    def look_down(cls, amount=1.0):
        super().look_down()
        cls._record_action(cls.LOOK_DOWN)

    @classmethod
    def look_left(cls, amount=1.0):
        super().look_left()
        cls._record_action(cls.LOOK_LEFT)

    @classmethod
    def look_right(cls, amount=1.0):
        super().look_right()
        cls._record_action(cls.LOOK_RIGHT)

    @classmethod
    def move_forward(cls, amount=1.0):
        super().move_forward()
        cls._record_action(cls.MOVE_FORWARD)

    @classmethod
    def shoot(cls, amount=1.0):
        super().shoot()
        cls._record_action(cls.SHOOT)

    @classmethod
    def do_action(cls, action_vector):
        if action_vector[cls.LOOK_UP]: cls().look_up()
        if action_vector[cls.LOOK_DOWN]: cls().look_down()
        if action_vector[cls.LOOK_LEFT]: cls().look_left()
        if action_vector[cls.LOOK_RIGHT]: cls().look_right()
        if action_vector[cls.MOVE_FORWARD]: cls().move_forward()
        if action_vector[cls.SHOOT]: cls().shoot()


class DrivingControls(Controls):
    ACTION_VECTOR = [0.0 for _ in range(3)]

    @classmethod
    def do_action(cls, action):
        if action is None:
            return

        h.Controls.set_control_normal(27, h.eControl.control_vehicle_move_left_right, action[0])
        h.Controls.set_control_normal(27, h.eControl.control_vehicle_accelerate, action[1])
        h.Controls.set_control_normal(27, h.eControl.control_vehicle_brake, action[2])
