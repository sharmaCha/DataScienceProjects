import tensorflow as tf
W = tf.Variable([.3],tf.float32)
b= tf.Variable([-.3],tf.float32)
x= tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variable_initialized()
sess=tf.Session()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))
y = tf.placeholder(tf.float32)
squared_deltas=tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

