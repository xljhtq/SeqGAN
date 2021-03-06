## encoding=utf8
import sys

import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import vocab_utils
import os

#########################################################################################
PRE_EPOCH_NUM_generator = 30  # supervised (maximum likelihood estimation) epochs
PRE_EPOCH_NUM_discriminator = 30
BATCH_SIZE = 10
need_generated_samples = 200000
TOTAL_BATCH = 50

g_lrn = 0.01
d_lrn = 0.0001
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 128  # embedding dimension
HIDDEN_DIM = 200  # hidden state dimension of lstm cell
SEQ_LENGTH = 20  # sequence length
START_TOKEN = 0
SEED = 88

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 128
dis_filter_sizes = [2, 3, 5]
dis_num_filters = [100, 100, 100]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.01

#########################################################################################
#  Basic Training Parameters
#########################################################################################
train_dir = "./"
log = open('data/experiment-log.txt', 'w')
positive_file = 'data/positive_data.txt'
negative_file = 'data/negative_data.txt'  ## generator model 生成的fake data
out_negative_file = 'data/negative_data_'  ## 输出观察的negative_data
eval_file = 'data/eval_data.txt'  ## generator model 生成的每次训练后的test集


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # using MLE
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def transform_file(negative_file, wordVocab, out_file):
    out_op = open(out_file, "w")
    for line in open(negative_file):
        line = line.strip("\n").split(" ")
        wordList = []
        for id in line:
            wordList.append(wordVocab.id2word[int(id)])
        out_op.write(" ".join(wordList) + "\n")
    out_op.close()


def main(source_file, wordVocab, vocab_size):
    tf.reset_default_graph()
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    dis_data_loader = Dis_dataloader(BATCH_SIZE)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing

    # todo:  print ("starting generating positive samples...")
    generated_num = gen_data_loader.transform_positive_file_2(train_dir + source_file,
                                                              train_dir + positive_file,
                                                              wordVocab,
                                                              SEQ_LENGTH)
    print ("generated_num: ", generated_num)
    if generated_num < 100: return
    gen_data_loader.create_batches(train_dir + positive_file)

    with tf.variable_scope("Train", reuse=None):
        generator = Generator(wordVocab,
                              vocab_size,
                              BATCH_SIZE,
                              EMB_DIM,
                              HIDDEN_DIM,
                              SEQ_LENGTH,
                              START_TOKEN,
                              learning_rate=g_lrn)

        discriminator = Discriminator(word_vocab=wordVocab,
                                      sequence_length=SEQ_LENGTH,
                                      num_classes=2,
                                      embedding_size=dis_embedding_dim,
                                      filter_sizes=dis_filter_sizes,
                                      num_filters=dis_num_filters,
                                      l2_reg_lambda=dis_l2_reg_lambda,
                                      learning_rate=d_lrn)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # todo:  1.##############pre-train generator##############
    print 'Start pre-training generator with MLE...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM_generator):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        if epoch % 5 == 0:
            buffer = 'epoch:\t' + str(epoch) + '\tloss:\t' + str(loss)
            print (buffer)
            sys.stdout.flush()
            log.write(buffer)
            # generate_samples(sess,
            #                  generator,
            #                  BATCH_SIZE,
            #                  generated_num,
            #                  eval_file)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            # print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            # buffer = 'epoch:\t' + str(epoch) + '\tnllscore:\t' + str(test_loss) + '\n'
            # log.write(buffer)

    # todo:  2.##############pre-train discriminator##############
    print 'Start pre-training discriminator...'
    for _ in range(PRE_EPOCH_NUM_discriminator):
        ## 由于是对概率分布的采样，所以每次生成的fake data数据都是不同的
        generate_samples(sess,
                         generator,
                         BATCH_SIZE,
                         generated_num,
                         negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for _ in range(3):  ## 对每批fake_data进行训练discriminator
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    with tf.variable_scope("Train", reuse=None):
        g_beta = ROLLOUT(generator, 0.8)  ## 这是表示 g_beta

    # todo:  3.############## Adversarial Training ##############
    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        # todo: Train the generator for one batch samples
        for it in range(1):
            samples = generator.generate(sess)
            rewards = g_beta.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples,
                    generator.rewards: rewards}
            _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)

        # Test
        if total_batch % 10 == 0 or total_batch == TOTAL_BATCH - 1:
            buffer = 'epoch:\t' + str(total_batch) + '\tg_loss:\t' + str(g_loss)
            print (buffer)
            sys.stdout.flush()
            log.write(buffer)

        g_beta.update_params()

        # todo: Train the discriminator
        for _ in range(5):
            generate_samples(sess,
                             generator,
                             BATCH_SIZE,
                             generated_num,
                             negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)

        if total_batch % 10 == 0 or total_batch == TOTAL_BATCH - 1:
            out_file = out_negative_file + str(total_batch) + ".txt"
            transform_file(negative_file, wordVocab, out_file)

    generate_samples(sess,
                     generator,
                     BATCH_SIZE,
                     need_generated_samples,
                     negative_file)
    transform_file(negative_file, wordVocab, source_file + ".GEN")


if __name__ == '__main__':
    path = "data/all_2/"
    fileList = os.listdir(path)
    print ("start loading vocab...")
    wordVocab = vocab_utils.Vocab()
    wordVocab.fromText_format3(train_dir, "data/wordvec.vec")
    vocab_size = wordVocab.vocab_size
    print ("vocab_size: ", vocab_size)

    for file in fileList:
        print (path + file)
        main(path + file, wordVocab, vocab_size)
