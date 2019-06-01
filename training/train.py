import os
import argparse
import collections

import torch
import torch.nn.functional as F
import numpy as np
import tqdm

import model
import resnet_pytorch
import datasets


TrainArguments = collections.namedtuple(
        'TrainArguments',
        [ 'desired', 'n_actions', 'network', 'loss',
            'n_scenarios', 'n_frames',
            'data_dir', 'save_dir', 'model_path', 'log_path',
            'batch_size', 'lr', 'n_decay', 'weight_decay', 'train_iterations',
            'n_workers', 'on_cluster', 'cuda'])
Log = collections.namedtuple(
        'Log',
        ['epoch', 'train_loss', 'test_loss'])
OUTPUTS = {
        'depth': 5,
        'label': 10,
        'flow': 6,
        'material': 4,
        }


def get_n_channels(desired):
    n_channels = 0
    n_channels += 4 * int('image' in desired)
    n_channels += 5 * int('depth' in desired)
    n_channels += 10 * int('label' in desired)
    n_channels += 6 * int('flow' in desired)
    n_channels += 4 * int('material' in desired)

    return n_channels


def get_l1_loss(y_hat, y):
    diff = torch.abs(torch.squeeze(y_hat) - torch.squeeze(y))
    loss = diff.mean()
    metrics = [loss.cpu().detach().item()]

    return loss, metrics


def get_xent_loss(y_hat, y):
    loss = torch.nn.functional.cross_entropy(y_hat, y)

    pred = torch.max(y_hat, 1)[1]
    accuracy = (pred == y).float().mean().item()

    return loss, [accuracy]


def get_bce_loss(y_hat, y):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
    accuracy = ((y_hat > 0).float() == y).float().mean().item()

    return loss, [accuracy]


def get_custom_loss(y_hat, y):
    y_hat_1 = y_hat[:,:1]
    y_1 = y[:,:1]

    bce_loss, bce_metrics = get_bce_loss(y_hat_1, y_1)

    y_hat_2 = y_hat[:,1:]
    y_2 = y[:,1:]

    l1_loss, l1_metrics = get_l1_loss(y_hat_2, y_2)

    loss = bce_loss + l1_loss
    metrics = [bce_loss.item()] + bce_metrics + [l1_loss.item()] + l1_metrics

    return loss, metrics


def setup(args):
    for field in args._fields:
        print('%s: %s' % (field, getattr(args, field)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_data, test_data = datasets.make_gtav_dataset(
            args.desired, args.n_actions, args.data_dir, args.batch_size, args.n_workers,
            True, args.n_scenarios, args.n_frames)

    print(len(train_data))
    print(len(test_data))

    if args.network == 'resnet':
        net = resnet_pytorch.resnet18(get_n_channels(args.desired), 1)
    elif args.network == 'nin':
        net = model.NetworkInNetwork(get_n_channels(args.desired), 1)

    optim = torch.optim.Adam(
            net.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay)

    if os.path.exists(args.model_path):
        net_state, optim_state = torch.load(args.model_path, map_location='cpu')

        net.load_state_dict(net_state)
        optim.load_state_dict(optim_state)

        print('Loaded successfully from %s.' % args.model_path)

    if os.path.exists(args.log_path):
        log = torch.load(args.log_path)
    else:
        log = Log(0, [], [])

    if args.cuda:
        net = net.cuda()

        for tensors in optim.state.values():
            for k, v in tensors.items():
                if isinstance(v, torch.Tensor):
                    tensors[k] = v.cuda()

    return train_data, test_data, net, optim, log


def find_mean_std(data):
    means = list()
    stds = list()

    for x, y in tqdm.tqdm(data, desc='Batch'):
        means.append(y.mean(0).detach().numpy())
        stds.append(y.std(0).detach().numpy())

    print(list(np.mean(means, 0)))
    print(list(np.mean(stds, 0)))

    import pdb; pdb.set_trace()

    return None


def train_or_test(net, data, optim, args):
    is_train = optim is not None

    if args.loss == 'l1':
        loss_func = get_l1_loss
    elif args.loss == 'xent':
        loss_func = get_xent_loss
    elif args.loss == 'custom':
        loss_func = get_custom_loss
    elif args.loss == 'unet':
        loss_func = get_unet_loss
    else:
        loss_func = None

    if is_train:
        net = net.train()
    else:
        net = net.eval()

    all_metrics = list()

    for x, y in tqdm.tqdm(data, desc='Batch', disable=args.on_cluster):
        # Hack. makes assumption about order. too lazy, will change.
        if args.loss == 'unet':
            y = x[:,4:-4]
            x = torch.cat([x[:,:4], x[:,-4:]], 1)

        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        y_hat = net(x)
        loss, metrics = loss_func(y_hat, y)

        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()

        all_metrics.append([loss.item()] + metrics)

    return all_metrics


def main(args):
    train_data, test_data, net, optim, log = setup(args)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            int(args.train_iterations // len(train_data) // (args.n_decay + 1)) + 1,
            gamma=0.5)

    for i in tqdm.tqdm(
            range(0, args.train_iterations, len(train_data)),
            desc='Epoch', disable=args.on_cluster):
        scheduler.step()

        if i == 0:
            train_metrics = train_or_test(net, train_data, None, args)
        else:
            train_metrics = train_or_test(net, train_data, optim, args)

        with torch.no_grad():
            test_metrics = train_or_test(net, test_data, None, args)

        to_print = [
                '%d / %d' % (i // len(train_data), args.train_iterations // len(train_data)),
                ' '.join(['%.2f' % x for x in  np.mean(train_metrics, 0)]),
                ' '.join(['%.2f' % x for x in  np.mean(test_metrics, 0)])]

        print('\t'.join(to_print))

        j_save = None

        # Save to checkpoint.
        for j in range(i, i + len(train_data)):
            if j % (args.train_iterations // 100) == 0:
                j_save = j

        if j_save is not None:
            log.train_loss.append(train_metrics)
            log.test_loss.append(test_metrics)
            log = log._replace(epoch=i)

            torch.save(log, args.log_path + '_%s' % j_save)
            torch.save((net.state_dict(), optim.state_dict()), args.model_path + '_%s' % j_save)

            print('Saved at iteration %d.' % j_save)


def parse():
    _bool = lambda x: str(x).lower() in ['y', '1']
    _list = lambda x: x.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--desired', type=_list, required=True)
    parser.add_argument('--n_actions', type=int, required=True)
    parser.add_argument('--network', type=str, default='nin')
    parser.add_argument('--loss', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--n_scenarios', type=int, default=10000)
    parser.add_argument('--n_frames', type=int, default=100000)
    parser.add_argument('--save_dir', type=str, default='saved')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_decay', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--train_iterations', type=int, default=100000)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--on_cluster', type=_bool, default=False)
    parser.add_argument('--cuda', type=_bool, default=torch.cuda.is_available())

    parsed = parser.parse_args()

    assert parsed.loss in ['l1', 'xent', 'custom', 'unet']

    data_dir = os.path.abspath(parsed.data_dir)
    save_dir = os.path.abspath(parsed.save_dir)

    return TrainArguments(
            parsed.desired,
            parsed.n_actions,
            parsed.network,
            parsed.loss,
            parsed.n_scenarios,
            parsed.n_frames,
            data_dir,
            save_dir,
            os.path.join(save_dir, 'model.t7'),
            os.path.join(save_dir, 'log.t7'),
            parsed.batch_size, parsed.lr, parsed.n_decay,
            parsed.weight_decay, parsed.train_iterations,
            parsed.n_workers, parsed.on_cluster, parsed.cuda)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    main(parse())
