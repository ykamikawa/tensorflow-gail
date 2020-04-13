import copy

import tensorflow as tf


class PPOTrain:
    """PPO学習用クラス"""

    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        # Policy network
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        # 学習可能な変数の取得
        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # インスタンス作成時にPolicyとOld_Policyのパラメータを同じにしておく
        with tf.variable_scope("assign_op"):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # 入力用のプレースホルダー定義
        with tf.variable_scope("train_inp"):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")
            self.rewards = tf.placeholder(
                dtype=tf.float32, shape=[None], name="rewards"
            )
            self.v_preds_next = tf.placeholder(
                dtype=tf.float32, shape=[None], name="v_preds_next"
            )
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name="gaes")

        # ある行動についてだけのprobを取得
        act_probs = self.Policy.act_probs
        act_probs = act_probs * tf.one_hot(
            indices=self.actions, depth=act_probs.shape[1]
        )
        act_probs = tf.reduce_sum(act_probs, axis=1)

        act_probs_old = self.Old_Policy.act_probs
        act_probs_old = act_probs_old * tf.one_hot(
            indices=self.actions, depth=act_probs_old.shape[1]
        )
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope("loss"):
            # ratios = tf.divide(act_probs, act_probs_old)
            # trust regionを計算 をpi(a|s)/pi_old(a|s)
            # 更新後の方策と更新前の方策のKL距離の制約を与える
            ratios = tf.exp(
                tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0))
            )
            # ratiosをclipping
            clipped_ratios = tf.clip_by_value(
                ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value
            )
            # clipping前とclipping後のlossで小さい方を使う
            loss_clip = tf.minimum(
                tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios)
            )
            loss_clip = tf.reduce_mean(loss_clip)
            # summaryにclipping lossを追加
            tf.summary.scalar("loss_clip", loss_clip)

            # 探索を促すためのentropy制約項
            # 方策のentropyが小さくなりすぎるのを防ぐ
            entropy = -tf.reduce_sum(
                self.Policy.act_probs
                * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)),
                axis=1,
            )
            entropy = tf.reduce_mean(entropy, axis=0)
            tf.summary.scalar("entropy", entropy)

            # 状態価値の分散を大きくしないための制約項
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(
                self.rewards + self.gamma * self.v_preds_next, v_preds
            )
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar("value_difference", loss_vf)

            # 以下の式を最大化
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy

            # tensorflowのoptimizerは最小最適化を行うため
            loss = -loss
            tf.summary.scalar("total", loss)

        # 全てのsummaryを取得するoperation
        self.merged = tf.summary.merge_all()

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        # 勾配の取得
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
        # train operation
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        """train operation実行関数"""
        tf.get_default_session().run(
            self.train_op,
            feed_dict={
                self.Policy.obs: obs,
                self.Old_Policy.obs: obs,
                self.actions: actions,
                self.rewards: rewards,
                self.v_preds_next: v_preds_next,
                self.gaes: gaes,
            },
        )

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next):
        """summary operation実行関数"""
        return tf.get_default_session().run(
            self.merged,
            feed_dict={
                self.Policy.obs: obs,
                self.Old_Policy.obs: obs,
                self.actions: actions,
                self.rewards: rewards,
                self.v_preds_next: v_preds_next,
                self.gaes: gaes,
            },
        )

    def assign_policy_parameters(self):
        """PolicyのパラメータをOld_Policyに代入"""
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        """
        GAE: generative advantage estimator
        Advantage関数で何ステップ先まで考慮すべきかを1つの式で表現
        rewards: 即時報酬系列
        v_preds: 状態価値
        v_preds_next: 次の状態の状態価値
        """
        # advantage関数
        # 現時点で予測している状態価値と実際に行動してみたあとの状態価値との差
        deltas = [
            r_t + self.gamma * v_next - v
            for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)
        ]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        # is T-1, where T is time step which run policy
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        """
        勾配計算関数
        obs: 状態
        actions: 行動系列
        gaes: generative advantage estimator
        rewards: 即時報酬系列
        v_preds_next: 次の状態価値関数
        """
        return tf.get_default_session().run(
            self.gradients,
            feed_dict={
                self.Policy.obs: obs,
                self.Old_Policy.obs: obs,
                self.actions: actions,
                self.rewards: rewards,
                self.v_preds_next: v_preds_next,
                self.gaes: gaes,
            },
        )
