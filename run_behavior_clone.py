import argparse
import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from network_models.policy_net import Policy_net
from algo.behavior_clone import BehavioralCloning


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='', default='trained_models/bc')
    parser.add_argument('--max_to_keep', help='', default=10, type=int)
    parser.add_argument('--logdir', help='', default='log/train/bc')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', help='', default=128, type=int)
    parser.add_argument('--epoch_num', help='', default=10, type=int)
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    # gym環境作成
    env = gym.make('CartPole-v0')

    # policy net作成
    Policy = Policy_net('policy', env)

    # BehavioralCloning学習インスタンス
    BC = BehavioralCloning(Policy)

    # エキスパートのtrajectory読み込み
    observations = np.genfromtxt('trajectory/observations.csv')
    actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    # tensorflow saver
    saver = tf.train.Saver()
    # session config
    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu_num,
                allow_growth=True
                ))

    # session
    with tf.Session(config=config) as sess:
        # summary writer
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        # Sessionの初期化
        sess.run(tf.global_variables_initializer())

        # エキスパートtrajectoryをpolicy net入力用に変換
        inp = [observations, actions]
        # episode loop
        for iteration in tqdm(range(args.iteration)):
            # epoch回だけoptimize
            for epoch in range(args.epoch_num):
                # 学習データをサンプリング
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=args.minibatch_size)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                # BehavioralCloningの学習
                BC.train(obs=sampled_inp[0], actions=sampled_inp[1])

            # summaryの取得
            summary = BC.get_summary(obs=inp[0], actions=inp[1])

            if (iteration+1) % args.interval == 0:
                saver.save(sess, args.savedir + '/model.ckpt', global_step=iteration+1)
            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
