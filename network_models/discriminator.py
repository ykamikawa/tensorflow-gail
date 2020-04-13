import tensorflow as tf


class Discriminator:
    def __init__(self, env):
        """
        env: 環境
        Generatice Adversarial by Imitation Learningのdiscriminatorクラス
        報酬関数の近似を行なっており,expertの方策は1,agentの方策は0になるようにtrain
        """

        with tf.variable_scope("discriminator"):
            # get vaiable scope name
            self.scope = tf.get_variable_scope().name
            # expert state placeholder
            self.expert_s = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(env.observation_space.shape)
            )
            # expert action placeholder
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)
            # add noise to expert action
            expert_a_one_hot += (
                tf.random_normal(
                    tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32
                )
                / 1.2
            )
            # concatenate state and action to input discriminator
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)

            # agent state placeholder
            self.agent_s = tf.placeholder(
                dtype=tf.float32, shape=[None] + list(env.observation_space.shape)
            )
            # agent action placeholder
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)
            # add noise to agent action
            agent_a_one_hot += (
                tf.random_normal(
                    tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32
                )
                / 1.2
            )
            # concatenate state and action to input discriminator
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope("network") as network_scope:
                expert_prob = self.construct_network(input=expert_s_a)
                # 同じスコープのnameをもつパラメータを共有
                network_scope.reuse_variables()
                agent_prob = self.construct_network(input=agent_s_a)

            with tf.variable_scope("loss"):
                # expertのrewardは大きく
                loss_expert = tf.reduce_mean(
                    tf.log(tf.clip_by_value(expert_prob, 0.01, 1))
                )
                # agentのrewardは小さくしたいので1から引く
                loss_agent = tf.reduce_mean(
                    tf.log(tf.clip_by_value(1 - agent_prob, 0.01, 1))
                )
                # tensorflowなので最小化 reward->cost
                loss = loss_expert + loss_agent
                loss = -loss
                # summaryにdiscriminatorのlossを追加
                tf.summary.scalar("discriminator", loss)

            # optimizer
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(loss)

            # 報酬関数固定でエージェントの報酬推定operation
            self.rewards = tf.log(tf.clip_by_value(agent_prob, 1e-10, 1))

    def construct_network(self, input):
        """
        input: expertかactionのstate-action
        discriminatorのbuild関数
        """
        layer_1 = tf.layers.dense(
            inputs=input, units=20, activation=tf.nn.leaky_relu, name="layer1"
        )
        layer_2 = tf.layers.dense(
            inputs=layer_1, units=20, activation=tf.nn.leaky_relu, name="layer2"
        )
        layer_3 = tf.layers.dense(
            inputs=layer_2, units=20, activation=tf.nn.leaky_relu, name="layer3"
        )
        # sigmoid activation 0~1
        prob = tf.layers.dense(
            inputs=layer_3, units=1, activation=tf.sigmoid, name="prob"
        )
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        """
        expert_s, expert_a: expertのstate-action
        agent_s, agent_a: agentのstate-action
        Discriminatorのtrain-step関数
        """
        return tf.get_default_session().run(
            self.train_op,
            feed_dict={
                self.expert_s: expert_s,
                self.expert_a: expert_a,
                self.agent_s: agent_s,
                self.agent_a: agent_a,
            },
        )

    def get_rewards(self, agent_s, agent_a):
        """
        agent_s: agentの状態
        agent_a: agentの行動
        方策固定でagentの報酬をdiscriminatorで推定する関数
        """
        return tf.get_default_session().run(
            self.rewards, feed_dict={self.agent_s: agent_s, self.agent_a: agent_a}
        )

    def get_trainable_variables(self):
        """学習可能なパラメータのみ取得する関数"""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
