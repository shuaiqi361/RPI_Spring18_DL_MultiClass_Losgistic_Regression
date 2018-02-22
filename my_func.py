import tensorflow as tf
import numpy as np


def compute_loss(w, data, label, lam):
    m, _ = data.shape
    prob, _ = compute_softmax(w, data)
    loss = - tf.reduce_sum(label * tf.log(prob), [0, 1]) + 0.5 * lam * tf.reduce_sum(w ** 2, [0, 1])
    return loss


def compute_softmax(w, data):
    # output a m * 5 softmax matrix and the argmax for each row
    exp_terms = tf.exp(tf.matmul(data, w))

    prob = exp_terms / tf.reduce_sum(exp_terms, axis=1, keepdims=True)
    max_ind = tf.argmax(prob, axis=1)
    
    return prob, max_ind


def compute_gradients(w, data, label, lam):

    prob, _ = compute_softmax(w, data)
    w_grad = tf.matmul(tf.transpose(data), (prob - label))
    reg_grad = w * lam

    return w_grad + reg_grad


