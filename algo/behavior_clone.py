import tensorflow as tf


class BehavioralCloning:
    '''BehavioralCloningクラス'''
    def __init__(self, Policy):
        self.Policy = Policy

        # エキスパートの行動
        self.actions_expert = tf.placeholder(tf.int32, shape=[None], name='actions_expert')
        actions_vec = tf.one_hot(self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32)

        # cross_entropy loss
        loss = tf.reduce_sum(actions_vec * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), 1)
        loss = tf.reduce_mean(loss)
        # lossのsummary
        tf.summary.scalar('loss/cross_entropy', loss)

        # 最適化
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(loss)

        # summaryの書き込み
        self.merged = tf.summary.merge_all()

    def train(self, obs, actions):
        return tf.get_default_session().run(
                self.train_op,
                feed_dict={
                    self.Policy.obs: obs,
                    self.actions_expert: actions})

    def get_summary(self, obs, actions):
        return tf.get_default_session().run(
                self.merged,
                feed_dict={
                    self.Policy.obs: obs,
                    self.actions_expert: actions})
