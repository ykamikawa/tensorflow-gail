import gym
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm

from network_models.policy_net import Policy_net


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='', default='trained_models')
    parser.add_argument('--alg', help='', default='gail')
    parser.add_argument('--model', help='', default='')
    parser.add_argument('--logdir', help='', default='log/test')
    parser.add_argument('--iteration', help='', default=int(1e3))
    parser.add_argument('--stochastic', help='', action='store_false')
    parser.add_argument('--gpu_num', help='specify GPU number', default='-1', type=str)
    return parser.parse_args()


def main(args):
    # prepare log dir
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # gym環境作成
    env = gym.make('CartPole-v0')
    # policy net
    Policy = Policy_net('policy', env)
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
        # 学習済みモデルの読み込み
        if args.model == '':
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+'model.ckpt')
        else:
            # モデル番号の選択
            saver.restore(sess, args.modeldir+'/'+args.alg+'/'+'model.ckpt-'+args.model)
        # 状態の初期化
        obs = env.reset()
        reward = 0
        success_num = 0

        # episode loop
        for iteration in tqdm(range(args.iteration)):
            rewards = []
            run_policy_steps = 0
            # run episode
            while True:
                run_policy_steps += 1
                # prepare to feed placeholder Policy.obs
                # ネットワーク入力用にobsを変換
                obs = np.stack([obs]).astype(dtype=np.float32)
                # 行動と価値を推定
                act, _ = Policy.act(obs=obs, stochastic=args.stochastic)

                # 要素数が1の配列をスカラーに変換
                act = np.asscalar(act)

                # episodeの各変数を追加
                rewards.append(reward)

                # policy netで推定した行動で状態の更新
                next_obs, reward, done, info = env.step(act)

                # episode終了判定
                # episodeが終了していたら次のepisodeを開始
                if done:
                    obs = env.reset()
                    reward = -1
                    break
                else:
                    obs = next_obs

            # summary追加
            writer.add_summary(
                    tf.Summary(
                        value=[tf.Summary.Value(
                            tag='episode_length',
                            simple_value=run_policy_steps)]),
                    iteration)
            writer.add_summary(
                    tf.Summary(
                        value=[tf.Summary.Value(
                            tag='episode_reward',
                            simple_value=sum(rewards))]),
                    iteration)

            # episode成功判定
            if sum(rewards) >= 195:
                success_num += 1
                # 連続で100回成功していればepisode loopを終了
                if success_num >= 100:
                    print('Iteration: ', iteration)
                    print('Clear!!')
                    break
            else:
                success_num = 0

        writer.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
