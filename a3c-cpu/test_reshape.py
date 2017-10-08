from PIL import Image
import gym
import helper
import numpy as np

env = gym.make('Breakout-v0')

s = env.reset()

img = helper.process_frame(s, 110*85, 110, 85)

img = np.reshape(img, [110, 85])

img = Image.fromarray(img, 'L')
img.show()
