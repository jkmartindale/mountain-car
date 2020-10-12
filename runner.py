import qlearning
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = qlearning.qlearningModel(0.9, 0.1, 5000, 40, 40)
shortRewards = []
mediumRewards = []
longRewards = []
shortMetrics = []
mediumMetrics = []
longMetrics = []
longScore = [0,0]
mediumScore = [0,0]
shortScore = [0,0]

for i in range(1):
    shortRewards, firstScore, bestScore = model.run(200)
    shortScore[0] += firstScore
    shortScore[1] += bestScore
    mediumRewards, firstScore, bestScore = model.run(500)
    mediumScore[0] += firstScore
    mediumScore[1] += bestScore
    longRewards, firstScore, bestScore= model.run(1000)
    longScore[0] += firstScore
    longScore[1] += bestScore


print("For 200 max steps, average best score: %d, average first episode under 200 steps: %d" % (shortScore[1]/10, shortScore[0]/10))
print("For 500 max steps, average best score: %d, average first episode under 200 steps: %d" % (mediumScore[1]/10, mediumScore[0]/10))
print("For 1000 max steps, average best score: %d, average first episode under 200 steps: %d" % (longScore[1]/10, longScore[0]/10))

episodes = np.arange(0,5000,1)
df = pd.DataFrame(shortRewards)
df2 = pd.DataFrame(mediumRewards)
df3 = pd.DataFrame(longRewards)

shortRollAve = df[0].rolling(window=100).mean()
mediumRollAve = df2[0].rolling(window=100).mean()
longRollAve = df3[0].rolling(window=100).mean()

#plt.plot(episodes, rewards)
plt.plot(episodes, shortRollAve, label='Max 200 steps per episode', color='red')
plt.plot(episodes, mediumRollAve, label='Max 500 steps per episode', color='blue')
plt.plot(episodes, longRollAve, label='Max 1000 steps per episode', color='green')
plt.legend(loc='upper left')
plt.xlabel("Episode")
plt.ylabel("Steps in Episode")
plt.show()

