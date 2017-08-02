import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def draw_traj(w_t, g_t, dw_t):
	plt.quiver(w_t[:,0], w_t[:,1], dw_t[:,0], dw_t[:,1], color='k')
	#plt.plot(w_t[:,0], w_t[:,1], '-*')




# Parameters for the intelligent synapse model
param_c = 1.0
param_xi = 0.1



learning_rate = 0.1



# A simple dummy model, corresponding to a 2D weight vector and a set of made-up loss functions
w_placeholder = tf.placeholder(tf.float32, (2,))
use_placeholder = tf.placeholder(tf.float32, (1,))

w_var = tf.Variable(0.1*tf.ones((2,)))
w = w_var*(1.0-use_placeholder) + (use_placeholder)*w_placeholder

el1 = tf.reduce_sum(w*tf.constant(np.asarray([1.,0.]),dtype=tf.float32))
el2 = tf.reduce_sum(w*tf.constant(np.asarray([0.,1.]),dtype=tf.float32))

loss1 = tf.square(el1-1.0)
loss2 = tf.square(el1-0.5)+tf.square(el2-0.5)



# Intelligent synapse
small_omega_var = tf.Variable(tf.zeros(w_var.get_shape()), trainable=False)
previous_weights_mu_minus_1 = tf.Variable(tf.zeros(w_var.get_shape()), trainable=False)
big_omega_var = tf.Variable(tf.zeros(w_var.get_shape()), trainable=False)

aux_loss = tf.reduce_sum(tf.multiply( big_omega_var, tf.square(previous_weights_mu_minus_1 - w_var) ))

reset_small_omega = tf.group( tf.assign( previous_weights_mu_minus_1, w_var ),  tf.assign( small_omega_var, small_omega_var*0.0 ) )

update_big_omega = tf.assign_add( big_omega_var,  tf.div(small_omega_var,(param_xi + tf.square(w_var-previous_weights_mu_minus_1) ))   )



optimizer = tf.train.GradientDescentOptimizer(learning_rate)

g1 = optimizer.compute_gradients(loss1, var_list=[w_var])
g1_a = optimizer.compute_gradients(loss1+aux_loss, var_list=[w_var])
train1 = optimizer.apply_gradients(g1_a)

g2 = optimizer.compute_gradients(loss2, var_list=[w_var])
g2_a = optimizer.compute_gradients(loss2+param_c*aux_loss, var_list=[w_var])
train2 = optimizer.apply_gradients(g2_a)


update_small_omega_1 = tf.assign_add( small_omega_var, learning_rate*g1_a[0][0]*g1[0][0] )
update_small_omega_2 = tf.assign_add( small_omega_var, learning_rate*g2_a[0][0]*g2[0][0] )





config = tf.ConfigProto()
config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())


## train and track weights and gradients
def training_util(g, train, update_small_omega, steps):
	weight_t = []
	gradient_t = []
	delta_w_t = []

	for t in range(steps):
		weight_t.append( sess.run(w, {w_placeholder:[0.0,0.0], use_placeholder:[0.0]}) )

		grad, _, _ = sess.run([g, train, update_small_omega], {w_placeholder:[0.0,0.0], use_placeholder:[0.0]})

		gradient_t.append( grad[0][0] )
		delta_w_t.append( -grad[0][0]*learning_rate ) # only valid for SGD!

	weight_t = np.asarray(weight_t)
	gradient_t = np.asarray(gradient_t)
	delta_w_t = np.asarray(delta_w_t)

	return weight_t, gradient_t, delta_w_t




weight_t_A, gradient_t_A, delta_w_t_A = training_util(g1, train1, update_small_omega_1, 15)

sess.run( update_big_omega )
sess.run( reset_small_omega )

weight_t_B, gradient_t_B, delta_w_t_B = training_util(g2, train2, update_small_omega_2, 15)





## compute the surface of each loss function
xmin=0.0
xmax=1.0
ymin=0.0
ymax=1.0
step = 0.05
x, y = np.meshgrid(np.arange(xmin, xmax+step, step), np.arange(ymin, ymax+step, step))
loss1_value = np.zeros(x.shape)
loss2_value = np.zeros(x.shape)

for i in xrange(x.shape[0]):
	for j in xrange(x.shape[1]):
		loss1_value[i,j],loss2_value[i,j] = sess.run([loss1, loss2], {w_placeholder:[x[i,j],y[i,j]], use_placeholder:[1.0]})




plt.subplot(1, 3, 1, aspect='equal')
plt.hold(True)
plt.title("Task 1")
plt.contour(x, y, loss1_value, 50)
draw_traj(weight_t_A, gradient_t_A, delta_w_t_A)
plt.hold(False)

plt.subplot(1, 3, 2, aspect='equal')
plt.hold(True)
plt.title("Task 2")
plt.contour(x, y, loss2_value, 50)
draw_traj(weight_t_B, gradient_t_B, delta_w_t_B)
plt.hold(False)


plt.subplot(1, 3, 3, aspect='equal')
plt.hold(True)
plt.title("Tasks 1+2 combined")
plt.contour(x, y, loss1_value+loss2_value, 50)
draw_traj(weight_t_A, gradient_t_A, delta_w_t_A)
draw_traj(weight_t_B, gradient_t_B, delta_w_t_B)
plt.hold(False)

plt.show()



