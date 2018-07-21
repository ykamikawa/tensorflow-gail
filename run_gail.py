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
    # 環境の作成
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
    # 入力: 観測, 出力: 行動の分布と収益の期待値のPolicy_net
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    # PPO学習用のインスタンス
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    # discriminator
    D = Discriminator(env)

    # エキスパートのtrajectories
    expert_observations = np.genfromtxt('trajectory/observations.csv')
    expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

    # 学習済みモデル保存用
    saver = tf.train.Saver()
    # sessoinの設定
    config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu_num,
                allow_growth=True
                ))

    with tf.Session(config=config) as sess:
        # summaryの準備
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        # session内の変数の初期化
        sess.run(tf.global_variables_initializer())
        # 環境の初期化
        obs = env.reset()
        reward = 0
        success_num = 0

        for iteration in tqdm(range(args.iteration)):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            run_policy_steps = 0
            # エピソードループ
            while True:
                run_policy_steps += 1
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
                rewards.append(reward)
                v_preds.append(v_pred)

                # 方策により決定した行動で環境を更新
                next_obs, reward, done, info = env.step(act)

                # エピソードが終了すれば環境とrewardをリセット
                if done:
                    v_preds_next = v_preds[1:] + [0]
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            # summaryの書き込み
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

            # 100回以上クリアできればiterarionを終了
            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, args.savedir + '/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            # Discriminatorの学習
            # 観測と行動をプレースホルダー用に変換
            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            # Discriminatorの学習ループ
            for i in range(2):
                D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)

            # Discriminatorの出力を報酬として利用
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            # d_rewardsを用いてgaesを取得
            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

            # d_rewardsを用いてPolicy_netの学習
            inp = [observations, actions, gaes, d_rewards, v_preds_next]
            # Old_PolicyにPolicy_netのパラメータを代入
            PPO.assign_policy_parameters()

            # PPOの学習
            for epoch in range(6):
                # indices are in [low, high)
                sample_indices = np.random.randint(
                        low=0,
                        high=observations.shape[0],
                        size=32)
                # 学習データをサンプル
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
