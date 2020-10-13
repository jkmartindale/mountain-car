import qlearning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = qlearning.qlearningModel(0.9, 0.2, 5000, 40, 40)
shortRewards = []

bestScoreList = []
firstScoreList = []
goodScoreList = []

for i in range(1):
    shortRewards, firstScore, bestScore, goodScore = model.run(1000)
    firstScoreList.append(firstScore)
    bestScoreList.append(bestScore)
    goodScoreList.append(goodScore)

print("For 200 max steps, average best score: %d, Standard Deviation: %d" % (np.mean(bestScoreList),
                                                                                                np.std(bestScoreList)))
print("For 200 max steps, average first under 200 steps episode: %d, Standard Deviation: %d" % (np.mean(firstScoreList),
                                                                                                np.std(firstScoreList)))
print("For 200 max steps, average under 200 steps: %d, Standard Deviation: %d" % (np.mean(goodScoreList),
                                                                                                np.std(goodScoreList)))


episodes = np.arange(0,5000,1)
df = pd.DataFrame(shortRewards)


shortRollAve = df[0].rolling(window=100).mean()


plt.plot(episodes, shortRewards)
plt.plot(episodes, shortRollAve, label='Max 200 steps per episode', color='red')

plt.legend(loc='upper left')
plt.xlabel("Episode")
plt.ylabel("Steps in Episode")
plt.show()

