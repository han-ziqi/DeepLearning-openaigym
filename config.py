# Maximum number of game rounds
maxEpisode = 3000
# Turn on visual reward thresholds
displayRewardThreshold = 100
# Save the model's rewards threshold
saveRewardThreshold = 100
# Maximum step length per turn
maxEpisodeStep = 2000
# Test Rounds
testEpisode = 20
# Whether to enable visualisation ( time consuming )
RENDER = False
# Decay coefficient of reward in TD error
GAMMA: float = 0.9
# Decay factor for running reward
runningRewardDecay=0.95
# Learning rate of the Actor network
actorLearningRate = 0.001
# Learning rate of Critic network
criticLearningRate = 0.01
# Number of neurons in FC layer
neuUnitNum = 20
# Number of seeds, reducing randomness
seed = 1
# Number of rounds to save the model
saveEpisodes = 100
# Model save path
model_dir = './models'