import qlearning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''Author: Andrew Swaim
   Date: 10/10/2020'''
'''Creates the model with the hyperparameter values'''
model = qlearning.qlearningModel(0.9, 0.2, 5000, 40, 40)
shortRewards = []
bestScoreList = []
firstScoreList = []
goodScoreList = []
'''Specifies max steps per episode'''
maxSteps = int(input("Enter max steps per episode: "))
numRuns = int(input("Enter number of trial runs: "))
'''Runs model multiple times and gets statistics'''
for i in range(1):
    shortRewards, firstScore, bestScore, goodScore = model.run(maxSteps)
    firstScoreList.append(firstScore)
    bestScoreList.append(bestScore)
    goodScoreList.append(goodScore)

print("For %d max steps, average best score: %d, Standard Deviation: %d" % (maxSteps, np.mean(bestScoreList),
                                                                                                np.std(bestScoreList)))
print("For %d max steps, average first under 200 steps episode: %d, Standard Deviation: %d" % (maxSteps, np.mean(firstScoreList),
                                                                                                np.std(firstScoreList)))
print("For %d max steps, average under 200 steps: %d, Standard Deviation: %d" % (maxSteps, np.mean(goodScoreList),
                                                                                                np.std(goodScoreList)))

'''Compiles values for graph and rolling mean'''
episodes = np.arange(0,5000,1)
df = pd.DataFrame(shortRewards)
shortRollAve = df[0].rolling(window=100).mean()

'''Plots reward values and rolling means'''
plt.plot(episodes, shortRewards)
plt.plot(episodes, shortRollAve, label='Max 200 steps per episode', color='red')

plt.legend(loc='upper left')
plt.xlabel("Episode")
plt.ylabel("Steps in Episode")
plt.show()

