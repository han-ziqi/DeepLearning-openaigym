
import tensorflow.compat.v1 as tf
import train
from config import model_dir, actorLearningRate, testEpisode, maxEpisodeStep


def modelTest():
    env, featuresNum, actionsNum = train.createEnv()
    sess = tf.Session()
    actor = train.Actor(sess, featureDimensions=featuresNum, actionSpace=actionsNum, learningRate=actorLearningRate)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    for i_episode in range(testEpisode):
        currentState = env.reset()
        currentStep = 0
        trackRecord = []
        while True:
            # Visualisation
            env.render()
            action = actor.chooseAction(currentState)
            nextState, reward, done, info = env.step(action)
            trackRecord.append(reward)
            currentState = nextState
            currentStep += 1

            if done or currentStep >= maxEpisodeStep:
                ep_rs_sum = sum(trackRecord)
                print("episode:", i_episode, "  reward:", int(ep_rs_sum), "  steps:", currentStep)
                break
modelTest()