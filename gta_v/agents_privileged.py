"""
These agents can directly interface with the game using anything in PyhookV.
"""
import time
import logging
import math

import numpy as np

import pyhookv as h
import pyhookv_utils as h_utils
import scenarios

import constants
import utils


class TaskStatus(object):
    SETTING_TASK = 'SETTING_TASK'
    WAITING_FOR_TASK_TO_FINISH = 'WAITING_FOR_TASK_TO_FINISH'


class TaskQueue(object):
    """
    Must override REQUIRED_DATA_KEYS.

    Queue logic:

    set_task
    while !is_task_done:
        do_backup
    """
    REQUIRED_DATA_KEYS = list()

    def __init__(self, set_task_funcs=None, is_task_done_funcs=None, do_backup_funcs=None, only_set_once=False):
        """
        if only_set_once is false, the set_task_func is called every tick.
        """
        self._set_task_funcs = set_task_funcs or self._make_set_task_funcs()
        self._is_task_done_funcs = is_task_done_funcs or self._make_is_task_done_funcs()

        self.only_set_once = only_set_once
        self.active_index = 0
        self._status = TaskStatus.SETTING_TASK
        self.status = constants.AgentStatus.RUNNING

        self._log_timer = utils.Timer(15.0)
        self._pause_timer = utils.Timer(0.1)

    def init(self, scenario):
        """
        Can override to get information about the scenario.
        """
        return

    def _make_set_task_funcs(self):
        """
        Must override if set_task_funcs not given to constructor.
        """
        return list()

    def _make_is_task_done_funcs(self):
        """
        Must override if is_task_done_funcs not given to constructor.
        """
        return list()

    def tick(self, data):
        """
        Returns last action.
        """
        if not self.has_valid_keys(data):
            return

        if self._pause_timer.is_time():
            h_utils.pause(0)

        if self._log_timer.is_time():
            logging.debug('Task: %s. Status: %s.' % (self.active_index, self._status))

        player_ped_id = h_utils.get_player_ped_id()

        set_task = self._set_task_funcs[self.active_index]
        is_task_done = self._is_task_done_funcs[self.active_index]

        # Reset action vector.
        h_utils.Controls.reset()

        if self.only_set_once:
            if self._status == TaskStatus.SETTING_TASK:
                set_task(**data)
                self._status = TaskStatus.WAITING_FOR_TASK_TO_FINISH
            elif self._status == TaskStatus.WAITING_FOR_TASK_TO_FINISH:
                if is_task_done(**data):
                    self.active_index = (self.active_index + 1) % len(self)
                    self._status = TaskStatus.SETTING_TASK
        else:
            set_task(**data)

            if is_task_done(**data):
                self.active_index = (self.active_index + 1) % len(self)

        return h_utils.Controls.get_last_action()

    def clear(self):
        self._status = TaskStatus.SETTING_TASK

        h.Ai.clear_ped_tasks_immediately(h_utils.get_player_ped_id())

    def has_valid_keys(self, data):
        missing = list()

        for key in self.REQUIRED_DATA_KEYS:
            if key not in data:
                missing.append(key)

        if not missing:
            return True

        logging.warn('Keys %s are missing from the data dict.' % list(sorted(missing)))
        logging.warn('State dict contains %s.' % list(sorted(data.keys())))

        return False

    def __len__(self):
        return len(self._set_task_funcs)


class WalkingAutopilot(TaskQueue):
    def __init__(self):
        super.__init__(
                [lambda **kwargs: h_utils.make_ped_wander(h_utils.get_player_ped_id())],
                [utils.always_returns_false], only_set_once=True)


class DrivingAutoPilot(TaskQueue):
    def __init__(self):
        """
        Real initialization is deferred until scenario.setup().
        """
        self.scenario = None

        self.drive_start = -1.0
        self.drive_time = 3.0

        self.noise_start = -1.0
        self.noise_time = 0.2

        self.status = constants.AgentStatus.NOT_STARTED

        super().__init__(only_set_once=True)

    def init(self, scenario):
        self.scenario = scenario

    def _make_set_task_funcs(self):
        result = list()
        result.append(self._drive)
        result.append(self._inject_noise)

        return result

    def _make_is_task_done_funcs(self):
        result = list()
        result.append(self._is_drive_done)
        result.append(self._is_noise_done)

        return result

    def _drive(self, **kwargs):
        if self.scenario is None:
            return

        # Hack.
        if hasattr(self.scenario, 'end_pos'):
            h.Ai.task_vehicle_drive_to_coord(
                    h_utils.get_player_ped_id(),
                    self.scenario.vehicle,
                    self.scenario.end_pos.x, self.scenario.end_pos.y, self.scenario.end_pos.z,
                    self.scenario.max_speed,
                    1,
                    self.scenario.vehicle_metadata.hash,
                    self.scenario.driving_style,
                    1.0, 0.0)
        else:
            h.Ai.task_vehicle_drive_wander(
                    h_utils.get_player_ped_id(),
                    self.scenario.vehicle,
                    self.scenario.max_speed,
                    self.scenario.driving_style)

        self.drive_start = time.time()

    def _is_drive_done(self, **kwargs):
        if self.drive_start < 0:
            return False

        if self.status == constants.AgentStatus.NOT_STARTED:
            self.status = constants.AgentStatus.RUNNING
        elif self.status == constants.AgentStatus.NOISE and \
                self.sign != np.sign(self.scenario.vehicle.control[0]) and \
                abs(self.scenario.vehicle.control[0]) > 0.1:
            self.status = constants.AgentStatus.RUNNING

        return (time.time() - self.drive_start) > self.drive_time

    def _inject_noise(self, **kwargs):
        self.status = constants.AgentStatus.NOISE

        self.angle = -abs(self.scenario.vehicle.control[0] * 5.0)
        self.throttle = self.scenario.vehicle.control[1]
        self.brake = self.scenario.vehicle.control[2]
        self.sign = int(self.angle < 0)

        h.Ai.clear_ped_tasks(h_utils.get_player_ped_id())

        self.noise_start = time.time()

    def _is_noise_done(self, **kwargs):
        self.scenario.vehicle.control = (self.angle, self.throttle, self.brake)

        if int(self.scenario.vehicle.is_damaged) == 1:
            self.scenario.vehicle.control = (0, 0, 0)
            return False

        if self.noise_start < 0:
            return False
        elif time.time() - self.noise_start > self.noise_time:
            return True

        return False

    def clear(self):
        """
        Keep player in vehicle after toggling autopilot.
        """
        if self.scenario is not None and self.scenario.vehicle is not None:
            h_utils.set_ped_into_vehicle(h_utils.get_player_ped_id(), self.scenario.vehicle)

        super().clear()

    def tick(self, data):
        super().tick(data)

        result = list(self.scenario.vehicle.control)
        result += [h_utils.get_vehicle_speed(self.scenario.vehicle)]

        return result


class ClientQueue(TaskQueue):
    REQUIRED_DATA_KEYS = list()

    def __init__(self):
        super().__init__()

        self.status = constants.AgentStatus.NOT_STARTED

    def _drive(self, data):
        a = data['action'][0]
        b = data['action'][1]
        c = data['action'][2]

        data['vehicle'].control = (a, b, c)

        if int(data['vehicle'].is_damaged) == 1:
            self.status = constants.AgentStatus.TERMINATED
        else:
            self.status = constants.AgentStatus.RUNNING

    def tick(self, data):
        if data['action'] is None or len(data['action']) == 0:
            return None

        self._drive(data)

        return data['action']


class Dummy(TaskQueue):
    REQUIRED_DATA_KEYS = list()

    def tick(self, data):
        return 123
