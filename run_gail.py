import os
import argparse
import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    # prepare log dir
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    # gym環境作成
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    # policy net
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    # ppo学習インスタンス
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    # discriminator
    D = Discriminator(env)

    # エキスパートtrajectory読み込み
    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    # tensorflow saver
    saver = tf.train.Saver()
    # session config
    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu_num,
                allow_growth=True
                ))
    # start session
    with tf.Session(config=config) as sess:
        # summary writer
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        # Sessionの初期化
        sess.run(tf.global_variables_initializer())
        # 状態の初期化
        obs = env.reset()
        success_num = 0
        # episode loop
        for iteration in tqdm(range(args.iteration)):
            # buffer
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            reward = 0
            # run episode
            while True:
                run_policy_steps += 1
                # ネットワーク入力用にobsを変換
                obs = np.stack([obs]).astype(dtype=np.float32)
                # 行動と価値を推定
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                # 要素数が1の配列をスカラーに変換
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                # policy netの推定行動で状態の更新
                next_obs, reward, done, info = env.step(act)

                # episodeの各変数を追加
                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                # episode終了判定
                if done:
                    v_preds_next = v_preds[1:] + [0]
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            # summary追加
            writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(
                        tag='episode_length',
                        simple_value=run_policy_steps)]),
                    iteration)
            writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(
                        tag='episode_reward',
                        simple_value=sum(rewards))]),
                    iteration)

            # episode成功判定
            if sum(rewards) >= 195:
                success_num += 1
                # 連続で100回成功していればepisode loopを終了
                if success_num >= 100:
                    saver.save(sess, args.savedir+'/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            # policy netによるtrajectryをプレースホルダー用に変換
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            ###########################
            # GAILの変更点はここだけ
            # discriminatorでエキスパートの報酬に近づける
            # discriminator学習
            for i in range(2):
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # discriminatorからpolicy netのrewardを取得
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            # rewardsをプレースホルダー用に変換
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
            ###########################

            # gaesの取得
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # PPO学習データ
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            # Old_Policyにパラメータを代入
            PPO.assign_policy_parameters()

            # PPOの学習
            for epoch in range(6):
                # 学習データサンプル用のインデックスを取得
                sample_indices = np.random.randint(
                        low=0,
                        high=observations.shape[0],
                        size=32)
                # PPO学習データをサンプル
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]
                PPO.train(
                        obs=sampled_inp[0],
                        actions=sampled_inp[1],
                        gaes=sampled_inp[2],
                        rewards=sampled_inp[3],
                        v_preds_next=sampled_inp[4])

            # summaryの取得
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
