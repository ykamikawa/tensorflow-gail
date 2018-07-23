import argparse
import gym
import numpy as np
import tensorflow as tf

from network_models.policy_net import Policy_net


def open_file_and_save(file_path, data):
    """
    csv保存用のindent checker
    file_path: type==string
    data:
    """
    try:
        with open(file_path, 'ab') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')
    except FileNotFoundError:
        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, data, fmt='%s')


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='filename of model to test', default='trained_models/ppo/model.ckpt')
    parser.add_argument('--iteration', default=10, type=int)
    parser.add_argument('--gpu_num', help='specify GPU number', default='0', type=str)
    return parser.parse_args()


def main(args):
    # gym環境作成
    env = gym.make('CartPole-v0')
    env.seed(0)
    ob_space = env.observation_space
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
        # Sessionの初期化
        sess.run(tf.global_variables_initializer())
        # 状態の初期化
        obs = env.reset()

        # エキスパートの学習済みモデルを読み込み
        saver.restore(sess, args.model)

        # episode
        for iteration in range(args.iteration):
            observations = []
            actions = []
            run_steps = 0
            while True:
                run_steps += 1
                # prepare to feed placeholder Policy.obs
                obs = np.stack([obs]).astype(dtype=np.float32)

                act, _ = Policy.act(obs=obs, stochastic=True)
                act = np.asscalar(act)

                observations.append(obs)
                actions.append(act)

                next_obs, reward, done, info = env.step(act)

                if done:
                    print(run_steps)
                    obs = env.reset()
                    break
                else:
                    obs = next_obs

            observations = np.reshape(observations, newshape=[-1] + list(ob_space.shape))
            actions = np.array(actions).astype(dtype=np.int32)

            open_file_and_save('trajectory/observations.csv', observations)
            open_file_and_save('trajectory/actions.csv', actions)


if __name__ == '__main__':
    args = argparser()
    main(args)
