# 24.07.'17
# jarvis
# adding an extra layer
#
#
# source: https://github.com/Hvass-Labs/TensorFlow-Tutorials

#%matplotlib inline #for jupyter
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

#load data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST",one_hot=True) # loaded one_hot encoded

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
t1 = time.time()

data.test.labels[0:5,:]
#convert one_hot to single number class
data.test.cls = np.array([label.argmax() for label in data.test.labels])
data.test.cls[0:5]

#We know that MNIST images are 28 pixels in each dimension
img_size = 28
#Images are stored in one-dimensional array of this length
img_size_flat = img_size * img_size
#Tuple with height and width of images used to reshape arrays
img_shape = (img_size,img_size)

#Number of classes, one class for each of 10 digits
num_classes = 10

#Helper function for plotting images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image. (reshape flat to 29x28)
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get the first images from the Test-set
images = data.test.images[0:9]

#Get the true classes for those images
cls_true = data.test.cls[0:9]

#Plot the images and labels using our helper function above
#plot_images(images=images, cls_true=cls_true)

# ====================== placeholder variables =========================

x = tf.placeholder(tf.float32,[None,img_size_flat])
#None determines number of images

#true labels
y_true = tf.placeholder(tf.float32,[None,num_classes])
#None determines number of labels
#one-hot - propability of matching a class

#true classes
y_true_cls = tf.placeholder(tf.int64,[None])

#number of first layer neurons
layer_1=200
layer_2=10
#number of seond layer neurons = num_classes

w1 = tf.Variable(tf.zeros([img_size_flat,layer_1]))
b1 = tf.Variable(tf.zeros([layer_1]))
w2 = tf.Variable(tf.zeros([layer_1,layer_2]))
b2 = tf.Variable(tf.zeros([layer_2]))

#model
Y1=tf.matmul(x,w1) + b1
Y2=tf.matmul(Y1,w2) + b2
#a matrix holding num_images rows and num_classes columns

y_pred = tf.nn.softmax(Y2)
y_pred_cls = tf.argmax(y_pred,dimension=1)

#one entropy per classified image
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y2,labels=y_true)
#average all entropies - single scalar value
cost = tf.reduce_mean(cross_entropy)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# ====================== run computational graph =========================

session =tf.Session()
session.run(tf.global_variables_initializer())




def optimize(num_iterations):
    #print()
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x:x_batch,
                       y_true: y_true_batch}
        session.run(optimizer,feed_dict_train)


feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}

def print_accuracy():
    acc = session.run(accuracy,feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(w1)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# ====================== compute =========================
batch_size = 1000
num_iterations = 1000
print("iterations: ",num_iterations)
optimize(num_iterations)
print_accuracy()
#plot_example_errors()
print("duration: ",time.time()-t1)
plot_weights()


session.close()
