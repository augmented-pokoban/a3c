import tensorflow as tf
import threading
import multiprocessing
import os
from time import sleep, time

from Network import Network
from Worker import Worker

max_episode_length = 301
max_buffer_length = 20
gamma = .99  # discount rate for advantage estimation and reward discounting
height = 110
width = 80
depth = 1
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = 4  # Agent can move Left, Right, or Fire
load_model = False
model_path = './model'
game_env = 'Breakout-v0'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
    master_network = Network(height, width, depth, s_size, a_size, 'global', None)  # Generate global network
    num_workers = multiprocessing.cpu_count()  # Set workers ot number of available CPU threads

    print 'Creating', num_workers, 'workers'
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(
            Worker(game_env, i, (height, width, depth, s_size), a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    if load_model:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, max_buffer_length)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
