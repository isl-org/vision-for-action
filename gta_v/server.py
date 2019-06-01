import time
import os

import numpy as np

import constants
import message_packer


class GTAVEnvironment(object):
    def __init__(self, blocking=False):
        self.blocking = int(blocking)

        self._server = message_packer.maybe_init_mmap(
                constants.SERVER_MMAP_PATH_LINUX, np.uint8, 'r+',
                message_packer.get_info_size(message_packer.SERVER_INFO))
        self._client = message_packer.maybe_init_mmap(
                constants.CLIENT_MMAP_PATH_LINUX, np.uint8, 'r+',
                message_packer.get_info_size(message_packer.CLIENT_INFO))

    def init(self, env_type=0, agent_type=0, ints=[], floats=[]):
        print('Connecting to server...')

        client_key = np.random.randint(1024)
        message = {
                'client_key': client_key,
                'timestamp': time.time(),
                'request_restart': 1,

                'agent_type': agent_type,
                'env_type': env_type,
                'scenario_params_int': ints,
                'scenario_params_float': floats,
                }

        message_packer.encode(self._client, message, message_packer.CLIENT_INFO)

        while self._get_server_data()['server_key'] != client_key:
            time.sleep(1)

        message = {
            'request_restart': 0,
            'client_key': -1,
            }

        message_packer.encode(self._client, message, message_packer.CLIENT_INFO)

        time.sleep(1)

        print('Server connection succeeded.')

        return True

    def step(self, action=None):
        if action is not None:
            message_packer.encode(self._client, {'actions_client': action}, message_packer.CLIENT_INFO)

        return self._get_server_data()

    def send(self, data):
        message_packer.encode(self._client, data, message_packer.CLIENT_INFO)

    def _get_server_data(self):
        return message_packer.decode(self._server, message_packer.SERVER_INFO)
