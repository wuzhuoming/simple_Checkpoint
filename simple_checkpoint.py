from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os


class Net(tf.keras.Model):
    """A simple linear model."""

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat(10).batch(2)


def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


checkpoint_name = 'ctl_step_{step}.ckpt'
model = Net()
opt = tf.keras.optimizers.Adam(0.1)

ckpt = tf.train.Checkpoint(optimizer=opt, model=model)

step = 0
for example in toy_dataset():
    loss = train_step(model, example, opt)
    step = step + 1
    if step % 10 == 0:
        checkpoint_path = os.path.join('C:\\Users\\Wu\\Desktop\\simple_tf_train_Checkpoint\\ckpt',
                                       checkpoint_name.format(step=step))
        save_path = ckpt.save(checkpoint_path)
        print("Saved checkpoint for step {}: {}".format(int(step), save_path))
        print("loss {:1.2f}".format(loss.numpy()))

lcf = tf.train.latest_checkpoint('C:\\Users\\Wu\\Desktop\\simple_tf_train_Checkpoint\\ckpt')
print("start restore")
ckpt.restore(lcf)
