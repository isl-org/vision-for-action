"""
Hotkeys -

F8: any script you want to run.
F9: toggles agent on/off.
F10: toggles both env and agent on/off.
F11: reloads all python files in the GTA V directory.
"""
import time
import logging
import io
import traceback

import numpy as np

# GameHook + PyHook imports.
import api
import pyhookv as h

import constants
import presets
import scenarios
import agents_privileged
import message_packer
import random_scripts
import utils; utils.setup_logging(logging.DEBUG)


AGENT_CLASS = {
        constants.AgentType.NONE: agents_privileged.Dummy,
        constants.AgentType.WALKING_AUTOPILOT: agents_privileged.WalkingAutopilot,
        constants.AgentType.DRIVING_AUTOPILOT: agents_privileged.DrivingAutoPilot,
        constants.AgentType.MANUAL: agents_privileged.ClientQueue,
        }


class PyPilot(api.BaseController):
    """
    Entry point into Gamehook.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Mmap locations.
        self._write = None
        self._read = None
        self._read_timer = utils.Timer(5)

        self._env = None
        self._agent = None

        # Zero out server.
        self._restart_server_state()

        # Set main loop.
        h.set_main_cb(self.loop_ignore_except)

    def _restart_server_state(self):
        self.toggle_agent(False)

        self.server_key = 1337
        self.can_start = False
        self.sync_mode = False

        self._is_env_running = False
        self._env_class = scenarios.DrivingPointGoal
        self._env = None

        self._is_agent_running = False
        self._agent_class = agents_privileged.Dummy
        self._agent = None
        self._agent_status = constants.AgentStatus.NOT_STARTED

        # Hack.
        self.scenario_params = presets.Driving.SCENE_1
        self.scenario_params_int = dict()
        self.scenario_params_float = dict()

        self.server_action = (0,)
        self.client_action = (0,)
        self.state = dict()
        self.targets = dict()

        self.prev_action_hash = float('inf')
        self._action_timer = utils.Timer(1.0 / 8.0)

        logging.info('Server state reset.')

    def update_buffers(self):
        if self.sync_mode:
            h.Gameplay.set_game_paused(0)

        for target in constants.TARGETS:
            self.targets[target] = self.get_target(target)

    def loop(self):
        if not self._update(first_call=True):
            return

        self._agent = self._agent_class()
        self._env = self._env_class(**{
            **self.scenario_params,
            **self.scenario_params_int,
            **self.scenario_params_float})

        self._env.start(self._agent)

        while self._update(first_call=False) and self._env.update():
            if self._is_agent_running:
                info = self._env.get_info()
                info['action'] = self.client_action

                self.server_action = self._agent.tick(info)
                self._agent_status = self._agent.status

            self.state = self._env.state()

    def toggle_agent(self, turn_on):
        if self._agent is None:
            return

        self._is_agent_running = turn_on
        self._agent.clear()

        logging.info('Agent %s.' % ((self._is_agent_running and 'running') or ('paused')))

    def on_key_down(self, key, special):
        if key == constants.F8:
            debug_callback()
        elif key == constants.F10:
            self._is_env_running = not self._is_env_running
            self.can_start = self._is_env_running
        elif key == constants.F11:
            self._restart_server_state()

        return False

    def _update(self, first_call=False):
        h.wait(0)

        self._maybe_init_mmap()

        if first_call and self.can_start:
            self._is_env_running = True
            self._is_agent_running = True
            self.can_start = False

            return True

        self._read_from_client()
        self._write_to_client()

        return self._is_env_running

    def _read_from_client(self):
        if self._read is None:
            return

        data = message_packer.decode(self._read, message_packer.CLIENT_INFO)

        self.sync_mode = (data['sync'] == 1)
        self.client_action = data['actions_client']

        # TODO(bradyz): implement better heartbeat.
        if data['request_restart'] and abs(data['timestamp'] - time.time()) < 10.0:
            if self.server_key != data['client_key']:
                self._restart_server_state()
                logging.info('%s' % data['client_key'])

                self.can_start = True
                self.scenario_params_int = {'ints': data['scenario_params_int']}
                self.scenario_params_float = {'floats': data['scenario_params_float']}
                self.server_key = data['client_key']
                return

        # TODO(bradyz): finish this up.
        if self.sync_mode:
            pass
        else:
            self.update_buffers()

    def _write_to_client(self):
        if self._write is None:
            return

        data = {
                'server_key': self.server_key,
                'timestamp': time.time(),

                'env_status': self._is_env_running,
                'agent_status': self._agent_status,

                'actions_agent': self.server_action,
                'reward': self.state.get('reward', np.nan),
                'reward_curve': self.state.get('reward_curve', np.nan),
                'position': self.state.get('position', (0.0, 0.0, 0.0))
                }

        for target in constants.TARGETS:
            data[target] = self.targets.get(target, 0)

        message_packer.encode(self._write, data, message_packer.SERVER_INFO)

    def unload(self):
        self._restart_server_state()

        if h.main_cb() == self.loop:
            h.set_main_cb(None)

        self._write = None
        self._read = None

    def _maybe_init_mmap(self):
        if self._write is None:
            self._write = message_packer.maybe_init_mmap(
                    constants.SERVER_MMAP_PATH, np.uint8, 'r+',
                    message_packer.get_info_size(message_packer.SERVER_INFO))

            if self._write is None:
                logging.warn('Could not create server.')
            else:
                logging.info('Allocated server memory.')

        if self._read is None and self._read_timer.is_time():
            self._read = message_packer.maybe_init_mmap(
                    constants.CLIENT_MMAP_PATH, np.uint8, 'r',
                    message_packer.get_info_size(message_packer.CLIENT_INFO))

            if self._read is None:
                logging.warn('No client connected.')
            else:
                logging.info('Connected to client.')

    def loop_ignore_except(self):
        try:
            self.loop()
        except Exception as e:
            f = io.StringIO()
            traceback.print_exc(file=f)

            api.warn('Scenario failed.', e)
            api.warn(f.getvalue())

        if self._agent is not None:
            self._agent.clear()
            self._agent = None

        if self._env is not None:
            self._env.stop()
            self._env = None


def debug_callback():
    """
    Run anything your heart desires.
    """
    random_scripts.change_env()
