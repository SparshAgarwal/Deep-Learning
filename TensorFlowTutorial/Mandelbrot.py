#TODO: Very inefficient implementation, the network keeps on calculating even though the value has diverged
#MAYBE: Instead of using a whole numpy array as one tensor node, split the whole array into nodes, once a node diverged, don't run it
import tensorflow as tf
import numpy as np
import PIL.Image
from io import BytesIO
from IPython.display import Image, display

def save_fractal_to_file(a, fmt='jpeg', filename='mandelbrot'):
	"""Display an array of iteration counts as a
		colorful picture of a fractal."""

	a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
	img = np.concatenate([10+20*np.cos(a_cyclic),
							30+50*np.sin(a_cyclic),
							155-80*np.cos(a_cyclic)], 2)
	img[a==a.max()] = 0
	a = img
	a = np.uint8(np.clip(a, 0, 255))
	f = BytesIO()
	PIL.Image.fromarray(a).save(filename + '.' + fmt, fmt, quality=100)

# Operation for computing Mandelbrot Set, Z = Z^2 + C
# NUM_ITERATION keeps track of the iteration needed for Z to grow larger than DIVERGE_CAP

# Change these values to make small changes to the image generated
DIVERGE_CAP = 4
NUM_ITERATION = 200
# Control how detailed you want the picture to be, the smaller the value, the more detailed but takes longer to run
RES = 0.001

# NP arrays
# Instead of using 0.005, use a smaller value for a more detailed image
Y, X = np.mgrid[-1.3:1.3:RES, -2:1:RES]
Z = X+1j*Y

# List to store image for every iteration
li = []

# TF variables
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

# Compute the new values of z: z^2 + x
zs_ = zs*zs + xs

# Have we diverged with this new value?
not_diverged = tf.abs(zs_) < DIVERGE_CAP

# Operation to update the zs and the iteration count.
#
# Note: We keep computing zs after they diverge! This
#       is very wasteful! There are better, if a little
#       less simple, ways to do this.
#
step = tf.group(
	zs.assign(zs_),
	ns.assign_add(tf.cast(not_diverged, tf.float32))
	)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(NUM_ITERATION): sess.run(step)

	save_fractal_to_file(sess.run(ns))
