import os
import argparse
import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from network_models.policy_net import Policy_net
from algo.ppo import PPOTrain


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    # 保存用ディレクトリの準備
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    # 環境のインスタンス
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    # 方策の更新前と更新後のpolicy networkの準備
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    # 学習ログの保存
    saver = tf.train.Saver()
    # sessoinの設定
    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu_num,
                allow_growth=True
                ))

    # session
    with tf.Session(config=config) as sess:
        # summaryの準備
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        # Sessionn内の変数の初期化
        sess.run(tf.global_variables_initializer())
        # 環境の初期化
        obs = env.reset()
        reward = 0
        success_num = 0

        # イテレーション開始
        for iteration in tqdm(range(args.iteration)):
            observations = []
            actions = []
            v_preds = []
            rewards = []
            episode_length = 0
            # エピソードループ
            while True:
                episode_length += 1
                # 観測をプレースホルダー用に変換
                obs = np.stack([obs]).astype(dtype=np.float32)
                # policy netに観測を入力し,行動と推定収益を取得
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                # 要素数が1の配列をスカラーに変換
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                # 現在の状態を追加
                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                # 方策により決定した行動で環境を更新
                next_obs, reward, done, info = env.step(act)

                if done:
                    # next state of terminate state has 0 state value
                    # エピソード終了時の状態の次の状態のvalueを0にする
                    v_preds_next = v_preds[1:] + [0]
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            # episodeのlog
            writer.add_summary(
                    tf.Summary(
                        value=[tf.Summary.Value(
                            tag='episode_length',
                            simple_value=episode_length)]),
                    iteration)
            # rewardsのlog
            writer.add_summary(
                    tf.Summary(
                        value=[tf.Summary.Value(
                            tag='episode_reward',
                            simple_value=sum(rewards))]),
                    iteration)

            # 収益が195を越えれば終了する
            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            PPO.assign_policy_parameters()

            inp = [observations, actions, gaes, rewards, v_preds_next]

            # train
            for epoch in range(6):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=32)
                # sample training data
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                PPO.train(
                        obs=sampled_inp[0],
                        actions=sampled_inp[1],
                        gaes=sampled_inp[2],
                        rewards=sampled_inp[3],
                        v_preds_next=sampled_inp[4])

            # PPOのsummaryの取得
            summary = PPO.get_summary(
                    obs=inp[0],
                    actions=inp[1],
                    gaes=inp[2],
                    rewards=inp[3],
                    v_preds_next=inp[4])

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
