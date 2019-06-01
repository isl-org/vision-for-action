import glob
import os

import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader, Dataset

import gta_v.recognition as sensors


def make_dataloader(dataset, is_train, batch_size, num_workers):
    return DataLoader(
                dataset, batch_size=batch_size,
                shuffle=is_train, num_workers=num_workers, drop_last=True)


def make_vizdoom_dataset(data_root, is_train, batch_size, num_workers):
    return make_dataloader(
            VizdoomDataset(data_root, is_train), is_train, batch_size, num_workers)


def make_gtav_dataset(desired, n_actions, data_root, batch_size, num_workers, preprocess,
        n_scenarios, n_frames):
    """
    Assumes directory structure
    data/
        inputs_000_00001.npy
        inputs_000_00002.npy
        ...
        action_000_00001.npy
        action_000_00002.npy
        ...
        roads_000.npy
        ...
        train.txt
        test.txt
    """
    def make_dataset(is_train):
        if is_train:
            filename = os.path.join(data_root, 'train.txt')
        else:
            filename = os.path.join(data_root, 'test.txt')

        with open(filename, 'r') as f:
            scenarios = list(map(int, f))

        first_n_scenarios = scenarios[:n_scenarios]
        inputs_actions = list()
        scenario_to_road_path = dict()

        for inputs_path in sorted(glob.glob(os.path.join(data_root, 'inputs_*.npy'))):
            tokens = os.path.basename(inputs_path).split('_')

            # HACK: Assumes /foo/bar/inputs_000_00001.npy.
            scenario = tokens[1]
            frame = tokens[2][:-4]

            if int(scenario) not in first_n_scenarios:
                continue
            if is_train and int(frame) > n_frames:
                continue
            if not is_train and int(frame) > 1000:
                continue

            action_path = os.path.join(data_root, 'action_%s_%s.npy' % (scenario, frame))
            road_path = os.path.join(data_root, 'labels_%s.npy' % scenario)

            if os.path.exists(action_path) and os.path.exists(road_path):
                scenario_to_road_path[scenario] = road_path
                inputs_actions.append((inputs_path, action_path))

        print(len(inputs_actions))

        return GTAVDataset(
                desired, n_actions,
                inputs_actions, scenario_to_road_path,
                is_train, preprocess)

    train_dataloader = make_dataloader(make_dataset(True), True, batch_size, num_workers)
    test_dataloader = make_dataloader(make_dataset(False), False, batch_size, num_workers)

    return train_dataloader, test_dataloader


class VizdoomDataset(Dataset):
    def __init__(self, data_root, is_train):
        self.paths = list(glob.glob(os.path.join(data_root, '*.npy')))
        self.is_train = is_train

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        data = np.transpose(data, (1, 2, 0))
        data = cv2.resize(data, (128, 128))

        # Have to do flip manually since Torch/ImgAug only supports for uint8.
        if self.is_train and np.random.rand() < 0.5:
            data = np.fliplr(data)

        data = np.transpose(data, (2, 0, 1))

        image = torch.FloatTensor(data[:1])
        others = torch.FloatTensor(data[1:])

        return image, others

    def __len__(self):
        return len(self.paths)


def get_scenario_number(filename):
    return os.path.basename(filename).split('_')[1].split('.')[0]


class GTAVDataset(Dataset):
    def __init__(self, desired, n_actions,
            inputs_action_paths, scenario_to_road_path,
            is_train, preprocess):
        self.desired = desired
        self.n_actions = n_actions
        self.inputs_action_paths = inputs_action_paths
        self.scenario_to_road_path = scenario_to_road_path

        self.is_train = is_train
        self.preprocess = preprocess

    def __getitem__(self, idx):
        """
        inputs_*.npy is saved as H x W x C.
        need to return as C x H x W.
        """
        road_ids = np.load(
                self.scenario_to_road_path[get_scenario_number(self.inputs_action_paths[idx][0])])

        inputs_all = np.load(self.inputs_action_paths[idx][0])
        inputs = sensors.get_inputs(self.desired, inputs_all, road_ids, self.preprocess)
        inputs = np.transpose(inputs, (2, 0, 1))
        inputs = torch.FloatTensor(inputs)

        actions = np.load(self.inputs_action_paths[idx][1])

        # HACK. Steering or Steering + throttle.
        if self.n_actions == 1:
            actions = np.float32([actions[0]])

            if self.preprocess:
                actions /= 0.27

            actions = torch.FloatTensor(actions)
        elif self.n_actions == 2:
            actions = np.float32([actions[0], actions[2]])

            if self.preprocess:
                actions -= (0.00, 0.004)
                actions /= (0.27, 0.028)

            actions = torch.FloatTensor(actions)

        return inputs, actions

    def __len__(self):
        return len(self.inputs_action_paths)
