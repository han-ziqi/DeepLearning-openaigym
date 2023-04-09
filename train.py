import os
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd

from config import seed, neuUnitNum, GAMMA, RENDER, criticLearningRate, maxEpisodeStep, runningRewardDecay, \
    saveEpisodes, model_dir, \
    maxEpisode, actorLearningRate


def createEnv():
    env = gym.make('LunarLander-v2')
    # Reduced randomness
    env.seed(seed)
    env = env.unwrapped
    featuresNum = env.observation_space.shape[0]
    actionsNum = env.action_space.n
    return env, featuresNum, actionsNum


class Actor(object):

    def __init__(self, sess, featureDimensions, actionSpace, learningRate=0.001):
        tf.compat.v1.disable_eager_execution()
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, featureDimensions], "state")
        self.a = tf.placeholder(tf.int32, None, "action")
        # TD_error
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=neuUnitNum,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=actionSpace,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            # Loss function
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            # minimize(-exp_v) = maximize(exp_v)
            self.train_op = tf.train.AdamOptimizer(learningRate).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def chooseAction(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):

    def __init__(self, sess, featureDimensions, learningRate=0.01) -> object:
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, featureDimensions], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                # number of hidden units
                units=neuUnitNum,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                # output units
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            # TD_error = (r+gamma*V_next) - V_eval
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learningRate).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


def modelTrain():
    env, featuresNum, actionsNum = createEnv()
    render = RENDER
    sess = tf.Session()
    actor = Actor(sess, featureDimensions=featuresNum, actionSpace=actionsNum, learningRate=actorLearningRate)
    critic = Critic(sess, featureDimensions=featuresNum, learningRate=criticLearningRate)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i_episode in range(maxEpisode + 1):
        currentState = env.reset()
        currentStep = 0
        trackRecord = []
        while True:
            action = actor.chooseAction(currentState)
            nextState, reward, done, info = env.step(action)
            trackRecord.append(reward)
            td_error = critic.learn(currentState, reward, nextState)
            actor.learn(currentState, action, td_error)
            currentState = nextState
            currentStep += 1
            if done or currentStep >= maxEpisodeStep:
                ep_rs_sum = sum(trackRecord)

                if 'runningReward' not in locals():
                    runningReward = ep_rs_sum
                else:
                    runningReward = runningReward * runningRewardDecay + ep_rs_sum * (1 - runningRewardDecay)
                # Determining whether a visualisation threshold has been reached
                print("episode:", i_episode, "  reward:", int(runningReward), "  steps:", currentStep)
                break
        if i_episode > 0 and i_episode % saveEpisodes == 0:
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            ckpt_path = os.path.join(model_dir, '{}_model.ckpt'.format(i_episode))
            saver.save(sess, ckpt_path)


# modelTrain()
# # reset graph
# tf.reset_default_graph()
