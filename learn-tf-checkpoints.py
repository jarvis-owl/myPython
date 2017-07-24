'''
    24.05.'17
    v01
    jarvis owl

    script to learn and understand TensorFlow

    source:
    https://www.tensorflow.org/programmers_guide/variables
    additional:
    http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

'''


import tensorflow as tf

#create some variables
#with tf.device("/gpu:0"): #do not know if this works correctly
weights = tf.Variable(tf.random_normal([784,200],stddev=0.035), name="weights")
biases = tf.Variable(tf.zeros([200]),name ="biases")
w2 = tf.Variable(weights.initialized_value() * 2, name="w2")
save_path = "/home/scout/github/myPython/model_container/model.ckpt"

#add an op to initialize variables
init_op = tf.global_variables_initializer()

#add ops to save and restore specified variables
saver = tf.train.Saver({"my_w2":w2})#or use a vector of variables [w2,weights,biases]
    #'max_to_keep', 'keep_checkpoint_every_n_hours' and 'latest_checkpoint()' possible

#later, when launching model
with tf.Session() as sess:
        #Run the init operation [neccessary, when not restoring ALL variables]
        sess.run(init_op)
        #load former variables
        saver.restore(sess,save_path)

        #save new processed variables
        saver.save(sess,save_path)
