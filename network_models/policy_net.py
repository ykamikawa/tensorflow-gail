import tensorflow as tf


class Policy_net:
    """方策ネットワーククラス"""

    def __init__(self, name: str, env):
        # 状態と行動空間の情報取得
        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            # 状態用のプレースホルダー
            self.obs = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(ob_space.shape), name="obs"
            )

            # 方策用ネットワーク
            with tf.variable_scope("policy_net"):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(
                    inputs=layer_2, units=act_space.n, activation=tf.tanh
                )
                self.act_probs = tf.layers.dense(
                    inputs=layer_3, units=act_space.n, activation=tf.nn.softmax
                )

            # 状態価値関数用ネットワーク
            with tf.variable_scope("value_net"):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            # 確率的に方策を決定
            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            # 決定的に方策を決定
            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            # ネットワークのscopeを取得
            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        """方策による行動決定関数"""
        if stochastic:
            return tf.get_default_session().run(
                [self.act_stochastic, self.v_preds], feed_dict={self.obs: obs}
            )
        else:
            return tf.get_default_session().run(
                [self.act_deterministic, self.v_preds], feed_dict={self.obs: obs}
            )

    def get_action_prob(self, obs):
        """方策ネットワークによるprob取得関数"""
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        """パラメータ取得関数"""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        """学習対象のパラメータ取得関数"""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
