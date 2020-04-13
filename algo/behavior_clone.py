import tensorflow as tf


class BehavioralCloning:
    """BehavioralCloning"""

    def __init__(self, Policy):
        # Policy network
        self.Policy = Policy

        # エキスパートの行動用のプレースホルダー(教師あり学習)
        self.actions_expert = tf.placeholder(
            tf.int32, shape=[None], name="actions_expert"
        )
        # onehot vectorに変換
        actions_vec = tf.one_hot(
            self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32
        )

        # cross_entropy lossを最小化
        loss = -tf.reduce_sum(
            actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1
        )
        loss = tf.reduce_mean(loss)
        # lossをsummaryに追加
        tf.summary.scalar("loss/cross_entropy", loss)

        # optimizer
        optimizer = tf.train.AdamOptimizer()
        # train operation
        self.train_op = optimizer.minimize(loss)

        # summary operation
        self.merged = tf.summary.merge_all()

    def train(self, obs, actions):
        """train operation実行関数"""
        return tf.get_default_session().run(
            self.train_op,
            feed_dict={self.Policy.obs: obs, self.actions_expert: actions},
        )

    def get_summary(self, obs, actions):
        """summary operation実行関数"""
        return tf.get_default_session().run(
            self.merged, feed_dict={self.Policy.obs: obs, self.actions_expert: actions}
        )
