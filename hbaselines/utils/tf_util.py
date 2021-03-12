"""TensorFlow utility methods."""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from functools import reduce

# Stabilizing term to avoid NaN (prevents division by zero or log of zero)
EPS = 1e-6


def make_session(num_cpu, graph=None):
    """Return a session that will use <num_cpu> CPU's only.

    Parameters
    ----------
    num_cpu : int
        number of CPUs to use for TensorFlow
    graph : tf.Graph
        the graph of the session

    Returns
    -------
    tf.compat.v1.Session
        a tensorflow session
    """
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)

    # Prevent tensorflow from taking all the gpu memory.
    tf_config.gpu_options.allow_growth = True

    return tf.compat.v1.Session(config=tf_config, graph=graph)


def get_trainable_vars(name=None):
    """Return the trainable variables.

    Parameters
    ----------
    name : str
        the scope

    Returns
    -------
    list of tf.Variable
        trainable variables
    """
    return tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def get_globals_vars(name=None):
    """Return the global variables.

    Parameters
    ----------
    name : str
        the scope

    Returns
    -------
    list of tf.Variable
        global variables
    """
    return tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=name)


def reduce_std(tensor, axis=None, keepdims=False):
    """Get the standard deviation of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor or tf.Variable
        the input tensor
    axis : int or list of int
        the axis to itterate the std over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """Get the variance of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    axis : int or list of int
        the axis to itterate the variance over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def var_shape(tensor):
    """
    get TensorFlow Tensor shape

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: ([int]) the shape
    """
    out = tensor.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(tensor):
    """
    get TensorFlow Tensor's number of elements

    :param tensor: (TensorFlow Tensor) the input tensor
    :return: (int) the number of elements
    """
    return int(np.prod(var_shape(tensor)))


def flatgrad(loss, var_list, clip_norm=None):
    """
    calculates the gradient and flattens it

    :param loss: (float) the loss value
    :param var_list: ([TensorFlow Tensor]) the variables
    :param clip_norm: (float) clip the gradients (disabled if None)
    :return: ([TensorFlow Tensor]) flattened gradient
    """
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[
        tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
        for (v, grad) in zip(var_list, grads)
    ])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32, sess=None):
        """
        Set the parameters from a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param dtype: (type) the type for the placeholder
        :param sess: (TensorFlow Session)
        """
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([int(np.prod(shape)) for shape in shapes])

        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, _var) in zip(shapes, var_list):
            size = int(np.prod(shape))
            assigns.append(tf.assign(
                _var, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.operation = tf.group(*assigns)
        self.sess = sess

    def __call__(self, theta):
        if self.sess is None:
            return tf.get_default_session().run(
                self.operation, feed_dict={self.theta: theta})
        else:
            return self.sess.run(
                self.operation, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list, sess=None):
        """
        Get the parameters as a flat vector

        :param var_list: ([TensorFlow Tensor]) the variables
        :param sess: (TensorFlow Session)
        """
        self.operation = tf.concat(
            axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        if self.sess is None:
            return tf.get_default_session().run(self.operation)
        else:
            return self.sess.run(self.operation)


def get_target_updates(_vars, target_vars, tau, verbose=0):
    """Get target update operations.

    Parameters
    ----------
    _vars : list of tf.Tensor
        the initial variables
    target_vars : list of tf.Tensor
        the target variables
    tau : float
        the soft update coefficient (keep old values, between 0 and 1)
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug

    Returns
    -------
    tf.Operation
        initial update
    tf.Operation
        soft update
    """
    if verbose >= 2:
        print('setting up target updates ...')

    soft_updates = []
    init_updates = []
    assert len(_vars) == len(target_vars)

    for var, target_var in zip(_vars, target_vars):
        if verbose >= 2:
            print('  {} <- {}'.format(target_var.name, var.name))
        init_updates.append(tf.compat.v1.assign(target_var, var))
        soft_updates.append(
            tf.compat.v1.assign(target_var, (1.-tau) * target_var + tau * var))

    assert len(init_updates) == len(_vars)
    assert len(soft_updates) == len(_vars)

    return tf.group(*init_updates), tf.group(*soft_updates)


def gaussian_likelihood(input_, mu_, log_std):
    """Compute log likelihood of a gaussian.

    Here we assume this is a Diagonal Gaussian.

    Parameters
    ----------
    input_ : tf.Variable
        the action by the policy
    mu_ : tf.Variable
        the policy mean
    log_std : tf.Variable
        the policy log std

    Returns
    -------
    tf.Variable
        the log-probability of a given observation given the output action
        from the policy
    """
    pre_sum = -0.5 * (((input_ - mu_) / (
                tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(
        2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu_, pi_, logp_pi):
    """Squash the output of the Gaussian distribution.

    This method also accounts for that in the log probability. The squashed
    mean is also returned for using deterministic actions.

    Parameters
    ----------
    mu_ : tf.Variable
        mean of the gaussian
    pi_ : tf.Variable
        output of the policy (or action) before squashing
    logp_pi : tf.Variable
        log probability before squashing

    Returns
    -------
    tf.Variable
        the output from the squashed deterministic policy
    tf.Variable
        the output from the squashed stochastic policy
    tf.Variable
        the log probability of a given squashed action
    """
    # Squash the output
    deterministic_policy = tf.nn.tanh(mu_)
    policy = tf.nn.tanh(pi_)

    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.math.log(1 - policy ** 2 + EPS), axis=1)

    return deterministic_policy, policy, logp_pi


def explained_variance(y_pred, y_true):
    """Compute fraction of variance that ypred explains about y.

    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    Parameters
    ----------
    y_pred : np.ndarray
        the prediction
    y_true : np.ndarray
        the expected value

    Returns
    -------
    float
        explained variance of ypred and y
    """
    var_y = reduce_var(y_true)
    return 1 - reduce_var(y_true - y_pred) / var_y


def print_params_shape(scope, param_type):
    """Print parameter shapes and number of parameters.

    Parameters
    ----------
    scope : str
        scope containing the parameters
    param_type : str
        the name of the parameter
    """
    shapes = [var.get_shape().as_list() for var in get_trainable_vars(scope)]
    nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in shapes])
    print('  {} shapes: {}'.format(param_type, shapes))
    print('  {} params: {}'.format(param_type, nb_params))


def layer(val,
          num_outputs,
          name,
          act_fun=None,
          kernel_initializer=slim.variance_scaling_initializer(
              factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
          layer_norm=False,
          batch_norm=False,
          phase=None,
          dropout=False,
          rate=None):
    """Create a fully-connected layer.

    Parameters
    ----------
    val : tf.Variable
        the input to the layer
    num_outputs : int
        number of outputs from the layer
    name : str
        the scope of the layer
    act_fun : tf.nn.* or None
        the activation function
    kernel_initializer : Any
        the initializing operation to the weights of the layer
    layer_norm : bool
        whether to enable layer normalization
    batch_norm : bool
        whether to enable batch normalization
    phase : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    dropout : bool
        whether to enable dropout
    rate : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented

    Returns
    -------
    tf.Variable
        the output from the layer
    """
    val = tf.layers.dense(
        val, num_outputs, name=name, kernel_initializer=kernel_initializer)

    if layer_norm:
        val = tf.contrib.layers.layer_norm(val, center=True, scale=True)

    if batch_norm:
        val = tf.contrib.layers.batch_norm(
            val,
            center=True,
            scale=True,
            is_training=phase,
            scope='bn_{}'.format(name),
        )

    if act_fun is not None:
        val = act_fun(val)

    if dropout:
        val = tf.nn.dropout(val, rate=rate)

    return val


def conv_layer(val,
               filters,
               kernel_size,
               strides,
               name,
               act_fun=None,
               kernel_initializer=slim.variance_scaling_initializer(
                   factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
               layer_norm=False,
               batch_norm=False,
               phase=None,
               dropout=False,
               rate=None):
    """Create a convolutional layer.

    Parameters
    ----------
    val : tf.Variable
        the input to the layer
    filters : int
        the number of channels in the convolutional kernel
    kernel_size : int or list of int
        the height and width of the convolutional filter
    strides : int or list of int
        the strides in each direction of convolution
    name : str
        the scope of the layer
    act_fun : tf.nn.* or None
        the activation function
    kernel_initializer : Any
        the initializing operation to the weights of the layer
    layer_norm : bool
        whether to enable layer normalization
    batch_norm : bool
        whether to enable batch normalization
    phase : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    dropout : bool
        whether to enable dropout
    rate : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented

    Returns
    -------
    tf.Variable
        the output from the layer
    """
    val = tf.layers.conv2d(
        val,
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        name=name,
        kernel_initializer=kernel_initializer
    )

    if layer_norm:
        val = tf.contrib.layers.layer_norm(val, center=True, scale=True)

    if batch_norm:
        val = tf.contrib.layers.batch_norm(
            val,
            center=True,
            scale=True,
            is_training=phase,
            scope='bn_{}'.format(name),
        )

    if act_fun is not None:
        val = act_fun(val)

    if dropout:
        val = tf.nn.dropout(val, rate=rate)

    return val


def create_fcnet(obs,
                 layers,
                 num_output,
                 stochastic,
                 act_fun,
                 layer_norm,
                 batch_norm,
                 phase,
                 dropout,
                 rate,
                 scope=None,
                 reuse=False,
                 output_pre=""):
    """Create a fully-connected neural network model.

    Parameters
    ----------
    obs : tf.Variable
        the input to the model
    layers : list of int
        the size of the neural network for the model
    num_output : int
        number of outputs from the model
    stochastic : bool
        whether the model should be stochastic or deterministic
    act_fun : tf.nn.* or None
        the activation function
    layer_norm : bool
        whether to enable layer normalization
    batch_norm : bool
        whether to enable batch normalization
    phase : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    dropout : bool
        whether to enable dropout
    rate : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented
    scope : str
        the scope name of the model
    reuse : bool
        whether or not to reuse parameters
    output_pre : str
        a string that is prepended to the name of the output layer. For
        backwards compatibility purposes.

    Returns
    -------
    tf.Variable or (tf.Variable, tf.Variable)
        the output from the model. a variable representing the output from the
        model in the deterministic case and a tuple of the (mean, logstd) in
        the stochastic case
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        pi_h = obs

        # Create the hidden layers.
        for i, layer_size in enumerate(layers):
            pi_h = layer(
                pi_h, layer_size, 'fc{}'.format(i),
                act_fun=act_fun,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                phase=phase,
                dropout=dropout,
                rate=rate,
            )

        if stochastic:
            # Create the output mean.
            policy_mean = layer(
                pi_h, num_output, 'mean',
                act_fun=None,
            )

            # Create the output log_std.
            log_std = layer(
                pi_h, num_output, 'log_std',
                act_fun=None,
            )

            policy = (policy_mean, log_std)
        else:
            # Create the output layer.
            policy = layer(
                pi_h, num_output, '{}output'.format(output_pre),
                act_fun=None,
                kernel_initializer=tf.random_uniform_initializer(
                    minval=-3e-3, maxval=3e-3)
            )

        return policy


def create_conv(obs,
                image_height,
                image_width,
                image_channels,
                ignore_flat_channels,
                ignore_image,
                filters,
                kernel_sizes,
                strides,
                act_fun,
                layer_norm,
                batch_norm,
                phase,
                dropout,
                rate,
                scope=None,
                reuse=False):
    """Create a convolutional network.

    Parameters
    ----------
    obs : tf.Variable
        the input to the model
    image_height : int
        the height of the image in the observation
    image_width : int
        the width of the image in the observation
    image_channels : int
        the number of channels of the image in the observation
    ignore_flat_channels : list
        channels of the proprioceptive state to be ignored
    ignore_image : bool
        observation includes an image but should it be ignored
    filters : list of int
        the number of channels in the convolutional kernel
    kernel_sizes : int or list of int
        the height and width of the convolutional filter
    strides : int or list of int
        the strides in each direction of convolution
    act_fun : tf.nn.* or None
        the activation function
    layer_norm : bool
        whether to enable layer normalization
    batch_norm : bool
        whether to enable batch normalization
    phase : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    dropout : bool
        whether to enable dropout
    rate : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented
    scope : str
        the scope name of the model
    reuse : bool
        whether or not to reuse parameters

    Returns
    -------
    tf.Variable
        the output from the network
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        batch_size = tf.shape(obs)[0]
        image_size = image_height * image_width * image_channels

        original_pi_h = obs
        pi_h = original_pi_h[:, image_size:]

        ignored_indx = [
            i for i in range(pi_h.shape[1]) if i not in ignore_flat_channels]

        if len(ignored_indx) > 0:
            pi_h_ignored = tf.gather(pi_h, ignored_indx, axis=1)

        # Ignoring the image is useful for the lower level for creating an
        # abstraction barrier.
        if not ignore_image:
            pi_h_image = tf.reshape(
                original_pi_h[:, :image_size],
                [batch_size, image_height, image_width, image_channels]
            )

            # Create the hidden convolutional layers.
            for i, (filter_i, kernel_size_i, stride_i) in enumerate(zip(
                    filters, kernel_sizes, strides)):
                pi_h_image = conv_layer(
                    pi_h_image,
                    filter_i,
                    kernel_size_i,
                    stride_i,
                    'conv{}'.format(i),
                    act_fun=act_fun,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    phase=phase,
                    dropout=dropout,
                    rate=rate,
                )

            h = pi_h_image.shape[1]
            w = pi_h_image.shape[2]
            c = pi_h_image.shape[3]
            pi_h = tf.concat(
                [tf.reshape(pi_h_image, [batch_size, h * w * c]) /
                 tf.cast(h * w * c, tf.float32),
                 pi_h], 1
            )
            if len(ignored_indx) > 0:
                pi_h = tf.concat([pi_h, pi_h_ignored], 1)

        return pi_h


def gae_returns(mb_rewards, mb_values, mb_dones, last_values, gamma, lam):
    """Compute the bootstrapped/discounted returns.

    Parameters
    ----------
    mb_rewards : array_like
        a minibatch of rewards from a given environment
    mb_values : array_like
        a minibatch of values computed by the policy from a given
        environment
    mb_dones : array_like
        a minibatch of done masks from a given environment
    last_values : array_like
        the value associated with the current observation within the
        environment
    gamma : float
        discount factor
    lam : float
        factor for trade-off of bias vs variance for Generalized Advantage
        Estimator

    Returns
    -------
    array_like
        GAE-style expected discounted returns.
    """
    n_steps = mb_rewards.shape[0]
    # print(mb_rewards)
    print(mb_dones)

    # Discount/bootstrap off value fn.
    mb_advs = np.zeros_like(mb_rewards)
    mb_vactual = np.zeros_like(mb_rewards)
    lastgaelam = 0
    traj_length = []
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            if mb_dones[-1]:
                traj_length.append(0)
            nextnonterminal = 1.0 - mb_dones[-1]
            nextvalues = last_values
            mb_vactual[t] = mb_rewards[t]
        else:
            if mb_dones[t + 1]:
                traj_length.append(0)
            nextnonterminal = 1.0 - mb_dones[t+1]
            nextvalues = mb_values[t+1]
            mb_vactual[t] = mb_rewards[t] \
                + gamma * nextnonterminal * nextvalues
        traj_length[-1] += 1
        delta = mb_rewards[t] \
            + gamma * nextvalues * nextnonterminal - mb_values[t]
        mb_advs[t] = lastgaelam = delta \
            + gamma * lam * nextnonterminal * lastgaelam
    mb_returns = mb_advs + mb_values
    print(traj_length)

    return mb_returns


def process_minibatch(mb_obs,
                      mb_contexts,
                      mb_actions,
                      mb_values,
                      mb_neglogpacs,
                      mb_all_obs,
                      mb_rewards,
                      mb_returns,
                      mb_dones,
                      last_values,
                      gamma,
                      lam,
                      num_envs,
                      max_traj_length):
    """Process a minibatch of samples.

    This method re-formats the data to numpy arrays that can be passed to
    the tensorflow placeholders, and computes the GAE terms

    Parameters
    ----------
    mb_obs : array_like
        a minibatch of observations
    mb_contexts : array_like
        a minibatch of contextual terms
    mb_actions : array_like
        a minibatch of actions
    mb_values : array_like
        a minibatch of estimated values by the policy
    mb_neglogpacs : array_like
        a minibatch of the negative log-likelihood of performed actions
    mb_all_obs : array_like
        a minibatch of full state observations (for multiagent envs)
    mb_rewards : array_like
        a minibatch of environment rewards
    mb_returns : array_like
        a minibatch of expected discounted returns
    mb_dones : array_like
        a minibatch of done masks
    last_values : array_like
        the value associated with the current observation within the
        environment
    gamma : float
        discount factor
    lam : float
        factor for trade-off of bias vs variance for Generalized Advantage
        Estimator
    num_envs : int
        number of environments used to run simulations in parallel.

    Returns
    -------
    array_like
        the reformatted minibatch of observations
    array_like
        the reformatted minibatch of contextual terms
    array_like
        the reformatted minibatch of actions
    array_like
        the reformatted minibatch of estimated values by the policy
    array_like
        the reformatted minibatch of the negative log-likelihood of
        performed actions
    array_like
        the reformatted minibatch of full state observations (for
        multiagent envs)
    array_like
        the reformatted minibatch of environment rewards
    array_like
        the reformatted minibatch of expected discounted returns
    array_like
        the reformatted minibatch of done masks
    array_like
        a minibatch of estimated advantages
    int
        the number of sampled steps in the minibatch
    """
    for env_num in range(num_envs):
        # Convert the data to numpy arrays.
        mb_obs[env_num] = np.concatenate(mb_obs[env_num], axis=0)
        mb_rewards[env_num] = np.asarray(mb_rewards[env_num])
        mb_actions[env_num] = np.concatenate(mb_actions[env_num], axis=0)
        mb_values[env_num] = np.concatenate(mb_values[env_num], axis=0)
        if mb_neglogpacs is not None:  # TRPO case
            mb_neglogpacs[env_num] = np.concatenate(
                mb_neglogpacs[env_num], axis=0)
        mb_dones[env_num] = np.asarray(mb_dones[env_num])

        # TODO
        if max_traj_length is not None:
            (mb_obs[env_num],
             mb_rewards[env_num],
             mb_actions[env_num],
             mb_values[env_num],
             _,
             mb_dones[env_num]) = segment_trajectory(
                mb_obs=mb_obs[env_num].copy(),
                mb_rewards=mb_rewards[env_num],
                mb_actions=mb_actions[env_num],
                mb_values=mb_values[env_num],
                mb_neglogpacs=None,
                mb_dones=mb_dones[env_num],
                max_traj_length=max_traj_length,
            )

        # Compute the bootstrapped/discounted returns.
        mb_returns[env_num] = gae_returns(
            mb_rewards=mb_rewards[env_num],
            mb_values=mb_values[env_num],
            mb_dones=mb_dones[env_num],
            last_values=last_values[env_num],
            gamma=gamma,
            lam=lam,
        )

    # TODO.
    mb_obs = [x for x in mb_obs if len(x) > 0]
    mb_rewards = [x for x in mb_rewards if len(x) > 0]
    mb_actions = [x for x in mb_actions if len(x) > 0]
    mb_values = [x for x in mb_values if len(x) > 0]
    mb_dones = [x for x in mb_dones if len(x) > 0]

    # Concatenate the stored data.
    if len(mb_obs) > 1:
        mb_obs = np.concatenate(mb_obs)
        mb_contexts = np.concatenate(mb_contexts, axis=0)
        mb_actions = np.concatenate(mb_actions, axis=0)
        mb_values = np.concatenate(mb_values, axis=0)
        if mb_neglogpacs is not None:  # TRPO case
            mb_neglogpacs = np.concatenate(mb_neglogpacs, axis=0)
        mb_all_obs = np.concatenate(mb_all_obs, axis=0)
        mb_returns = np.concatenate(mb_returns, axis=0)
    else:
        mb_obs = mb_obs[0]
        mb_contexts = mb_contexts[0]
        mb_actions = mb_actions[0]
        mb_values = mb_values[0]
        if mb_neglogpacs is not None:  # TRPO case
            mb_neglogpacs = mb_neglogpacs[0]
        mb_all_obs = mb_all_obs[0]
        mb_returns = mb_returns[0]

    # Compute the advantages.
    advs = mb_returns - mb_values
    mb_advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    n_steps = mb_obs.shape[0]

    return mb_obs, mb_contexts, mb_actions, mb_values, mb_neglogpacs, \
        mb_all_obs, mb_rewards, mb_returns, mb_dones, mb_advs, n_steps


def segment_trajectory(mb_obs,
                       mb_rewards,
                       mb_actions,
                       mb_values,
                       mb_neglogpacs,
                       mb_dones,
                       max_traj_length):
    """TODO.

    This method performs the following operations:

    1. TODO
    2. TODO
    3. TODO

    Parameters
    ----------
    mb_obs : TODO
        TODO
    mb_rewards : TODO
        TODO
    mb_actions : TODO
        TODO
    mb_values : TODO
        TODO
    mb_neglogpacs : TODO
        TODO
    mb_dones : TODO
        TODO
    max_traj_length : int
        TODO

    Returns
    -------
    TODO
        TODO
    TODO
        TODO
    TODO
        TODO
    TODO
        TODO
    TODO
        TODO
    TODO
        TODO
    """
    # Compute the length of each trajectory. The length of the trajectories are
    # defined as the number of time steps before a done mask is set to True.
    n_steps = mb_rewards.shape[0]
    traj_length = [0]
    t_start = [0]
    for t in range(n_steps - 1):
        traj_length[-1] += 1
        if mb_dones[t]:
            traj_length.append(0)
            t_start.append(t+1)
    t_start.append(n_steps)

    # Remove the trajectories whose length is length than the desired max
    # trajectory length.
    valid_trajectories = []
    for i in range(len(traj_length)):
        if traj_length[i] >= max_traj_length:
            valid_trajectories.extend(list(range(t_start[i], t_start[i+1])))

    mb_obs = mb_obs[valid_trajectories]
    mb_rewards = mb_rewards[valid_trajectories]
    mb_actions = mb_actions[valid_trajectories]
    mb_values = mb_values[valid_trajectories]
    if mb_neglogpacs is not None:
        mb_neglogpacs = mb_neglogpacs[valid_trajectories]
    mb_dones = mb_dones[valid_trajectories]

    # ======================================================================= #
    # Segment the remaining trajectories to set of subset trajectories that   #
    # are the desired length.                                                 #
    # ======================================================================= #

    # Recompute the trajectory start locations.
    n_steps = len(mb_dones)
    t_start = [0]
    for t in range(n_steps - 1):
        if mb_dones[t]:
            t_start.append(t + 1)
    t_start.append(n_steps)

    # Get the indices for sub-trajectories.
    indices = []
    for i in range(len(t_start) - 1):
        indx = t_start[i]
        while t_start[i + 1] - indx > max_traj_length:
            indices.append(list(range(indx, indx + max_traj_length)))
            indx += max_traj_length
        if indx > 0:
            indices.append(list(
                range(t_start[i + 1] - max_traj_length, t_start[i + 1])))

    # Get sub-trajectories.
    if len(indices) > 0:
        new_mb_obs = []
        new_mb_rewards = []
        new_mb_actions = []
        new_mb_values = []
        new_mb_neglogpacs = []
        new_mb_dones = []
        for indx in indices:
            new_mb_obs.extend(list(mb_obs[indx]))
            new_mb_rewards.extend(list(mb_rewards[indx]))
            new_mb_actions.extend(list(mb_actions[indx]))
            new_mb_values.extend(list(mb_values[indx]))
            if mb_neglogpacs is not None:
                new_mb_neglogpacs.extend(list(mb_neglogpacs[indx]))
            new_mb_dones.extend([0] * (max_traj_length - 1) + [1])

        # Convert to array
        mb_obs = np.array(new_mb_obs)
        mb_rewards = np.array(new_mb_rewards)
        mb_actions = np.array(new_mb_actions)
        mb_values = np.array(new_mb_values)
        if mb_neglogpacs is not None:
            mb_neglogpacs = np.array(new_mb_neglogpacs)
        mb_dones = np.array(new_mb_dones)

    return mb_obs, mb_rewards, mb_actions, mb_values, mb_neglogpacs, mb_dones


def setup_target_updates(model_scope, target_scope, scope, tau, verbose):
    """Create the soft and initial target updates.

    The initial model parameters are assumed to be stored under the scope name
    "model", while the target policy parameters are assumed to be under the
    scope name "target".

    If an additional outer scope was provided when creating the policies, they
    can be passed under the `scope` parameter.

    Parameters
    ----------
    model_scope : str
        the scope of the model parameters
    target_scope : str
        the scope of the target parameters
    scope : str or None
        the outer scope, set to None if not available
    tau : float
        target update rate
    verbose : int
        the verbosity level: 0 none, 1 training information, 2 tensorflow debug

    Returns
    -------
    tf.Operation
        initial target updates, to match the target with the model
    tf.Operation
        soft target update operations
    """
    if scope is not None:
        model_scope = scope + '/' + model_scope
        target_scope = scope + '/' + target_scope

    return get_target_updates(
        get_trainable_vars(model_scope),
        get_trainable_vars(target_scope),
        tau, verbose)
