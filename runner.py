import qlearning

short_run = qlearning.qlearningModel(0.9, 0.6, 1000, 2000, 40, 40, False)
short_run.run()