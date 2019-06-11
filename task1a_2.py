import numpy as np
import tensorflow as tf
import time
from import_pkl import *
from cnn_model import cnn
from dense import *
from save_result import *

# 超参数以及系统配置
mode = 'test'
max_step = 10000
batch_size = 128
test_size = 2880
sample_size = 500
l2_constant = 0.001
ini_learning_rate = 0.01  # 初始学习速率时0.1
decay_rate = 0.98  # 衰减率
decay_steps = 100

keep_prob = 0.5
display_step = 50
sample_step = display_step

checkpoint_ori_dir = 'CheckPoint'
exp_symbol = 'exp8'

# 训练
def train(batch_size):
    
    global global_step
    sess.run(tf.global_variables_initializer())
    start_time = time.time()
    top3_score = [0, 0, 0]
    top3_step = [0, 0, 0]
    top3_list = ['0', '0', '0']
    load_time = time.time()
    input_all_feat, input_all_label = LoadData_momery(mode)
    print('All time: ', time.time() - load_time)

    for global_step in range(max_step):
        rnd_indices = np.random.randint(0, 11520, batch_size)
        input_feat = input_all_feat[rnd_indices]
        input_label = input_all_label[rnd_indices]
        e_loss, train_loss, _, pred_output, accu = sess.run([error_loss, loss, optimizer, output, accuracy],
                                                      feed_dict={feat: input_feat, label: input_label, kp: keep_prob,
                                                                 bs: batch_size, training:True})
        accu = np.average(accu)
        if global_step % display_step == 0:
            print('Step is: %d | Time is: %f | error loss: %f | loss is: %f  |  accu:%f' % (
            global_step, time.time() - start_time, e_loss, train_loss, accu))


        if global_step % display_step == 1:
            SaveModel(checkpoint_ori_dir, exp_symbol, saver, sess, global_step)
            score = test(sample_size)
            comp(score, global_step, top3_score, top3_step, top3_list)
            print(top3_list)
            f = open('top3.txt', 'w')
            f.writelines(top3_list)
            f.close()

# 测试
def test(test_size):
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    #if LoadModel(checkpoint_ori_dir, exp_symbol, saver, sess) == True:
    #    print('Load Success!')
    #else:
    #    print('Load Failure')
    checkpoint_path = './CheckPoint/exp8/exp8-8551'
    saver.restore(sess, checkpoint_path)
    print('Load Success!')
    #rnd_indices = np.random.randint(0, 2880 , test_size)
    rnd_indices = np.arange(0, test_size)
    input_feat = test_input_all_feat[rnd_indices]
    input_label = test_input_all_label[rnd_indices]

    pred_output, softmax_output, test_loss, accu = sess.run([pred, softmax, error_loss, accuracy],
                                      feed_dict={feat: input_feat, label: input_label, kp: 1, bs: test_size, training:False})
    accu = np.average(accu)
    print('Time is: %f  | Accu: %f' % (time.time() - start_time, accu))
    print('-------------------------------------------------------------------')

    label_output = np.argmax(input_label, 1)
    print(pred_output[0:30])
    print(label_output[0:30])
    #pred_output = np.argmax(pred_output, 1)
    csvName = 'dev_pred.csv'
    softName = 'dev_softmax.csv'
    saveResult_to_csv(pred_output, softmax_output, csvName, softName)

    return accu

# 模型
sess = tf.Session()
feat = tf.placeholder(tf.float32, shape=[None, 563, 59, 2], name='input_feat')
label = tf.placeholder(tf.float32, shape=[None, 10], name='label')
kp = tf.placeholder(tf.float32, shape=None, name='keep_prob')
bs = tf.placeholder(tf.int32, shape=[], name='batch_size')
training = tf.placeholder(tf.bool, shape= None, name= 'training')
cnn_output = cnn(feat, bs, istraining=training)
output = dense_model2(cnn_output, 576, kp)
error_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label))
softmax = tf.nn.softmax(output)
pred = tf.argmax(softmax,1)
accuracy = tf.equal(tf.argmax(softmax,1), tf.argmax(label,1))

global_step = tf.Variable(0, name = 'global_step', trainable = False)

var_list = tf.trainable_variables()

g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
var_list = var_list + [g for g in g_list if 'global_step' in g.name]

test_input_all_feat, test_input_all_label = LoadData_momery('test')

for var in var_list:
    print(var)
if mode == 'train':
    l2_list = [var for var in var_list if 'gamma' not in var.name and 'beta' not in var.name
               and 'moving_mean' not in var.name and 'moving_variance' not in var.name and 'global_step' not in var.name]
    l2_loss = l2_constant * tf.reduce_sum([tf.nn.l2_loss(l2_var) for l2_var in l2_list])
    loss = error_loss + l2_loss
    #loss = error_loss
else:
    loss = error_loss

#learning_rate = tf.train.exponential_decay(ini_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
learning_rate = 0.001
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): #保证train_op在update_ops执行之后再执行。
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)

saver = tf.train.Saver(var_list, max_to_keep=200)
# continue_train()
if mode == 'train':
    train(batch_size)
else:
    test(test_size)



