
import tensorflow as tf

#tried to silent tf - unsuccessfully
#export TF_CPP_MIN_LOG_LEVEL=2
#tf.logging.set_verbosity(tf.logging.ERROR)



x1 = tf.constant([5])
x2 = tf.constant([6])


result = tf.multiply(x1,x2)
print (result)


sess = tf.Session()
print(sess.run(result))
sess.close
