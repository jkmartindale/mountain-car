import qlearning
import numpy as np
import matplotlib.pyplot as plt
import plotting

short_run = qlearning.qlearningModel(0.9, 0.6, 1000, 2000, 40, 40)
rewards = short_run.run()
episodes = np.arange(0,2000,1)
plt.plot(episodes, rewards)
plt.xlabel = "Episode"
plt.ylabel = "Steps in Episode"
plt.show()

