import time
import tensorflow as tf
import numpy as np
import os
import csv
import random
import gym
from collections import deque

from hbaselines.utils.tf_util import make_session
from hbaselines.utils.tf_util import get_globals_vars
from hbaselines.utils.tf_util import get_trainable_vars

import stable_baselines.common.tf_util as tf_util


def iterbatches(arrays,
                *,
                num_batches=None,
                batch_size=None,
                shuffle=True,
                include_final_partial_batch=True):
    """
    Iterates over arrays in batches, must provide either num_batches or
    batch_size, the other must be None.

    :param arrays: (tuple) a tuple of arrays
    :param num_batches: (int) the number of batches, must be None is batch_size
        is defined
    :param batch_size: (int) the size of the batch, must be None is num_batches
        is defined
    :param shuffle: (bool) enable auto shuffle
    :param include_final_partial_batch: (bool) add the last batch if not the
        same size as the batch_size
    :return: (tuples) a tuple of a batch of the arrays
    """
    assert (num_batches is None) != (batch_size is None), \
        'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n_samples = arrays[0].shape[0]
    assert all(a.shape[0] == n_samples for a in arrays[1:])
    inds = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(inds)
    sections = np.arange(0, n_samples, batch_size)[1:] \
        if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)


def traj_segment_generator(policy, env, horizon):
    """
    Compute target value using TD(lambda) estimator, and advantage with
    GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :return: (dict) generator that returns a dict with the following keys:
        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted
          reward)
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
    """
    # Initialize state variables
    step = 0
    # not used, just so we have the datatype
    action = env.action_space.sample()
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    episode_start = True  # marks if we're on first timestep of an episode

    while True:
        action, vpred = policy.step(
            observation.reshape(-1, *observation.shape))
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "total_timestep": current_it_len,
                    'continue_training': True
            }
            _, vpred = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(
                action, env.action_space.low, env.action_space.high)

        observation, reward, done, info = env.step(clipped_action[0])

        rewards[i] = reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                cur_ep_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            current_ep_len = 0
            observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE
    (lambda)

    :param seg: (dict) the current segment of the trajectory (see
        traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if
    # last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = \
            rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = \
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def conjugate_gradient(f_ax,
                       b_vec,
                       cg_iters=10,
                       verbose=False,
                       residual_tol=1e-10):
    """Calculate the conjugate gradient of Ax = b.

    Based on https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel
    p 312.

    Parameters
    ----------
    f_ax : function
        The function describing the Matrix A dot the vector x (x being the
        input parameter of the function)
    b_vec : array_like
        vector b, where Ax = b
    cg_iters : int
        the maximum number of iterations for converging
    verbose : bool
        print extra information
    residual_tol : float
        the break point if the residual is below this value

    Returns
    -------
    array_like
        vector x, where Ax = b
    """
    first_basis_vect = b_vec.copy()  # the first basis vector
    residual = b_vec.copy()  # the residual
    x_var = np.zeros_like(b_vec)  # vector x, where Ax = b
    residual_dot_residual = residual.dot(residual)  # L2 norm of the residual

    fmt_str = "%10i %10.3g %10.3g"
    title_str = "%10s %10s %10s"
    if verbose:
        print(title_str % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if verbose:
            print(fmt_str % (i, residual_dot_residual, np.linalg.norm(x_var)))
        z_var = f_ax(first_basis_vect)
        v_var = residual_dot_residual / first_basis_vect.dot(z_var)
        x_var += v_var * first_basis_vect
        residual -= v_var * z_var
        new_residual_dot_residual = residual.dot(residual)
        mu_val = new_residual_dot_residual / residual_dot_residual
        first_basis_vect = residual + mu_val * first_basis_vect

        residual_dot_residual = new_residual_dot_residual
        if residual_dot_residual < residual_tol:
            break

    if verbose:
        print(fmt_str %
              (cg_iters, residual_dot_residual, np.linalg.norm(x_var)))
    return x_var


class TRPO(object):
    """
    Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)

    :param policy: (ActorCriticPolicy or str) The policy model to use
        (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if
        registered in Gym, can be str)
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch
        (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient
        calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param verbose: (int) the verbosity level: 0 none, 1 training information,
        2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the
        creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the
        policy on creation
    :param seed: (int) Seed for the pseudo-random generators (python, numpy,
        tensorflow)
    """

    def __init__(self,
                 policy,
                 env,
                 gamma=0.99,
                 timesteps_per_batch=1024,
                 max_kl=0.01,
                 cg_iters=10,
                 lam=0.98,
                 entcoeff=0.0,
                 cg_damping=1e-2,
                 vf_stepsize=3e-4,
                 vf_iters=3,
                 verbose=0,
                 _init_setup_model=True,
                 policy_kwargs=None,
                 seed=None):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.seed = seed

        num_envs = 1

        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.assign_old_eq_new = None
        self.get_flat = None
        self.set_from_flat = None

        self._info_keys = []

        # init
        self.graph = None
        self.policy_tf = None
        self.sess = None
        self.summary = None
        self.episode_step = [0 for _ in range(num_envs)]
        self.episodes = 0
        self.steps = 0
        self.epoch_episode_steps = []
        self.epoch_episode_rewards = []
        self.epoch_episodes = 0
        self.epoch = 0
        self.episode_rew_history = deque(maxlen=100)
        self.episode_reward = [0 for _ in range(num_envs)]
        self.info_at_done = {key: deque(maxlen=100) for key in self._info_keys}
        self.info_ph = {}
        self.rew_ph = None
        self.rew_history_ph = None
        self.eval_rew_ph = None
        self.eval_success_ph = None
        self.saver = None

        # Create the model variables and operations.
        if _init_setup_model:
            self.trainable_vars = self.setup_model()

    def setup_model(self):
        np.set_printoptions(precision=3)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            self.sess = make_session(num_cpu=3, graph=self.graph)

            # Construct network for new policy
            self.policy_tf = self.policy(
                sess=self.sess,
                ob_space=self.observation_space,
                ac_space=self.action_space,
                co_space=None,
                verbose=self.verbose,
                learning_rate=1e-3,
                model_params=dict(
                    model_type="fcnet",
                    layers=[256, 256],
                    layer_norm=False,
                    batch_norm=False,
                    dropout=False,
                    act_fun=tf.nn.relu,
                    ignore_flat_channels=[],
                    ignore_image=False,
                    image_height=32,
                    image_width=32,
                    image_channels=3,
                    filters=[16, 16, 16],
                    kernel_sizes=[5, 5, 5],
                    strides=[2, 2, 2],
                ),
                n_minibatches=1e-3,
                n_opt_epochs=1e-3,
                gamma=1e-3,
                lam=1e-3,
                ent_coef=1e-3,
                vf_coef=1e-3,
                max_grad_norm=1e-3,
                cliprange=1e-3,
                cliprange_vf=1e-3,
                l2_penalty=1e-3,
                scope=None,
                num_envs=1,
            )

            with tf.variable_scope("loss", reuse=False):
                kloldnew = self.policy_tf.kl()
                ent = self.policy_tf.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                entbonus = self.entcoeff * meanent

                # advantage * pnew / pold
                ratio = tf.exp(
                    self.policy_tf.logp(self.policy_tf.action_ph, old=False) -
                    self.policy_tf.logp(self.policy_tf.action_ph, old=True))
                surrgain = tf.reduce_mean(ratio * self.policy_tf.advs_ph)

                optimgain = surrgain + entbonus
                self.losses = [optimgain, meankl, entbonus, surrgain, meanent]

                all_var_list = get_trainable_vars("model")
                var_list = [
                    v for v in all_var_list
                    if "/vf" not in v.name and "/q/" not in v.name]
                vf_var_list = [
                    v for v in all_var_list
                    if "/pi" not in v.name and "/logstd" not in v.name]

                self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                self.set_from_flat = tf_util.SetFromFlat(
                    var_list, sess=self.sess)

                klgrads = tf.gradients(meankl, var_list)
                shapes = [var.get_shape().as_list() for var in var_list]
                start = 0
                tangents = []
                for shape in shapes:
                    var_size = tf_util.intprod(shape)
                    tangents.append(tf.reshape(
                        self.policy_tf.flat_tangent[start: start + var_size],
                        shape))
                    start += var_size
                gvp = tf.add_n(
                    [tf.reduce_sum(grad * tangent)
                     for (grad, tangent) in zip(klgrads, tangents)])
                # Fisher vector products
                self.fvp = tf_util.flatgrad(gvp, var_list)

                self.assign_old_eq_new = tf.group(*[
                    tf.assign(oldv, newv) for (oldv, newv) in
                    zip(get_globals_vars("oldpi"), get_globals_vars("model"))])

                # Create the value function optimizer.
                vferr = tf.reduce_mean(tf.square(
                    self.policy_tf.value_flat - self.policy_tf.ret_ph))
                optimizer = tf.compat.v1.train.AdamOptimizer(self.vf_stepsize)
                self.vf_optimizer = optimizer.minimize(
                    vferr,
                    var_list=vf_var_list,
                )

                # Initialize the model parameters and optimizers.
                with self.sess.as_default():
                    self.sess.run(tf.compat.v1.global_variables_initializer())
                    self.policy_tf.initialize()

                th_init = self.get_flat()
                self.set_from_flat(th_init)

            self.grad = tf_util.flatgrad(optimgain, var_list)

        return get_trainable_vars("model") + get_trainable_vars("oldpi")

    def learn(self, total_timesteps):

        with self.sess.as_default():
            seg_gen = traj_segment_generator(
                self.policy_tf,
                self.env,
                self.timesteps_per_batch,
            )

            episodes_so_far = 0
            timesteps_so_far = 0
            iters_so_far = 0
            t_start = time.time()

            while timesteps_so_far < total_timesteps:

                print("********* Iteration %i ***********" % iters_so_far)

                print("Optimizing Policy...")

                seg = seg_gen.__next__()
                self._train(seg)

                # lr: lengths and rewards
                lens, rews = (seg["ep_lens"], seg["ep_rets"])
                self.epoch_episode_steps = lens
                self.epoch_episode_rewards = rews
                self.epoch_episodes = len(rews)
                self.episode_rew_history.extend(rews)
                episodes_so_far += len(lens)
                current_it_timesteps = seg["total_timestep"]
                timesteps_so_far += current_it_timesteps
                iters_so_far += 1
                self.steps += current_it_timesteps
                self.episodes += len(rews)

                self._log_training(file_path=None, start_time=t_start)

                self.epoch += 1

    def _train(self, seg):

        def fisher_vector_product(vec):
            return self.sess.run(self.fvp, feed_dict={
                self.policy_tf.flat_tangent: vec,
                self.policy_tf.obs_ph: fvpargs[0],
                self.policy_tf.action_ph: fvpargs[1],
                self.policy_tf.advs_ph: fvpargs[2],
            }) + self.cg_damping * vec

        add_vtarg_and_adv(seg, self.gamma, self.lam)
        atarg, tdlamret = seg["adv"], seg["tdlamret"]

        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-8)

        args = seg["observations"], seg["actions"], atarg
        # Subsampling: see p40-42 of John Schulman thesis
        # http://joschu.net/docs/thesis.pdf
        fvpargs = [arr[::5] for arr in args]

        self.sess.run(self.assign_old_eq_new)

        # run loss backprop with summary, and save the metadata (memory,
        # compute time, ...)
        grad, *lossbefore = self.sess.run(
            [self.grad] + self.losses,
            feed_dict={
                self.policy_tf.obs_ph: seg["observations"],
                self.policy_tf.action_ph: seg["actions"],
                self.policy_tf.advs_ph: atarg,
                self.policy_tf.ret_ph: tdlamret,
            }
        )

        lossbefore = np.array(lossbefore)
        if np.allclose(grad, 0):
            print("Got zero gradient. not updating")
        else:
            stepdir = conjugate_gradient(
                fisher_vector_product,
                grad,
                cg_iters=self.cg_iters,
                verbose=self.verbose >= 1,
            )
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
            # abs(shs) to avoid taking square root of negative values
            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
            fullstep = stepdir / lagrange_multiplier
            expectedimprove = grad.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = self.get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                self.set_from_flat(thnew)
                mean_losses = surr, kl_loss, *_ = self.sess.run(
                    self.losses,
                    feed_dict={
                        self.policy_tf.obs_ph: seg["observations"],
                        self.policy_tf.action_ph: seg["actions"],
                        self.policy_tf.advs_ph: atarg,
                    }
                )
                improve = surr - surrbefore
                print("Expected: %.3f Actual: %.3f" % (
                    expectedimprove, improve))
                if not np.isfinite(mean_losses).all():
                    print("Got non-finite value of losses -- bad!")
                elif kl_loss > self.max_kl * 1.5:
                    print("violated KL constraint. shrinking step.")
                elif improve < 0:
                    print("surrogate didn't improve. shrinking step.")
                else:
                    print("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                print("couldn't compute a good step")
                self.set_from_flat(thbefore)

        for _ in range(self.vf_iters):
            for (mbob, mbret) in iterbatches(
                    (seg["observations"], seg["tdlamret"]),
                    include_final_partial_batch=False,
                    batch_size=128,
                    shuffle=True):
                self.sess.run(self.vf_optimizer, feed_dict={
                    self.policy_tf.obs_ph: mbob,
                    self.policy_tf.action_ph: seg["actions"],
                    self.policy_tf.ret_ph: mbret,
                })

    def _log_training(self, file_path, start_time):
        """Log training statistics.

        Parameters
        ----------
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        start_time : float
            the time when training began. This is used to print the total
            training time.
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            # Rollout statistics.
            'rollout/episodes': self.epoch_episodes,
            'rollout/episode_steps': np.mean(self.epoch_episode_steps),
            'rollout/return': np.mean(self.epoch_episode_rewards),
            'rollout/return_history': np.mean(self.episode_rew_history),

            # Total statistics.
            'total/epochs': self.epoch + 1,
            'total/steps': self.steps,
            'total/duration': duration,
            'total/steps_per_second': self.steps / duration,
            'total/episodes': self.episodes,
        }

        # Information passed by the environment.
        combined_stats.update({
            'info_at_done/{}'.format(key): np.mean(self.info_at_done[key])
            for key in self.info_at_done.keys()
        })

        # Save combined_stats in a csv file.
        if file_path is not None:
            exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                w = csv.DictWriter(f, fieldnames=combined_stats.keys())
                if not exists:
                    w.writeheader()
                w.writerow(combined_stats)

        # Print statistics.
        print("-" * 67)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<30} | {:<30} |".format(key, val))
        print("-" * 67)
        print('')


# =========================================================================== #
#                                    Policy                                   #
# =========================================================================== #

"""TRPO-compatible feedforward policy."""

from hbaselines.base_policies import Policy
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import create_conv


class FeedForwardPolicy(Policy):

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 co_space,
                 verbose,
                 learning_rate,
                 model_params,
                 n_minibatches,
                 n_opt_epochs,
                 gamma,
                 lam,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 cliprange,
                 cliprange_vf,
                 l2_penalty,
                 scope=None,
                 num_envs=1):
        """Instantiate the policy object.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the current TensorFlow session
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
        co_space : gym.spaces.*
            the context space of the environment
        verbose : int
            the verbosity level: 0 none, 1 training information, 2 tensorflow
            debug
        l2_penalty : float
            L2 regularization penalty. This is applied to the policy network.
        model_params : dict
            dictionary of model-specific parameters. See parent class.
        learning_rate : float
            the learning rate
        n_minibatches : int
            number of training minibatches per update
        n_opt_epochs : int
            number of training epochs per update procedure
        gamma : float
            the discount factor
        lam : float
            factor for trade-off of bias vs variance for Generalized Advantage
            Estimator
        ent_coef : float
            entropy coefficient for the loss calculation
        vf_coef : float
            value function coefficient for the loss calculation
        max_grad_norm : float
            the maximum value for the gradient clipping
        cliprange : float or callable
            clipping parameter, it can be a function
        cliprange_vf : float or callable
            clipping parameter for the value function, it can be a function.
            This is a parameter specific to the OpenAI implementation. If None
            is passed (default), then `cliprange` (that is used for the policy)
            will be used. IMPORTANT: this clipping depends on the reward
            scaling. To deactivate value function clipping (and recover the
            original PPO implementation), you have to pass a negative value
            (e.g. -1).
        """
        super(FeedForwardPolicy, self).__init__(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            co_space=co_space,
            verbose=verbose,
            l2_penalty=l2_penalty,
            model_params=model_params,
            num_envs=num_envs,
        )

        self.learning_rate = learning_rate
        self.n_minibatches = n_minibatches
        self.n_opt_epochs = n_opt_epochs
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf

        # Create variables to store on-policy data.
        self.mb_rewards = [[] for _ in range(num_envs)]
        self.mb_obs = [[] for _ in range(num_envs)]
        self.mb_contexts = [[] for _ in range(num_envs)]
        self.mb_actions = [[] for _ in range(num_envs)]
        self.mb_values = [[] for _ in range(num_envs)]
        self.mb_neglogpacs = [[] for _ in range(num_envs)]
        self.mb_dones = [[] for _ in range(num_envs)]
        self.mb_all_obs = [[] for _ in range(num_envs)]
        self.mb_returns = [[] for _ in range(num_envs)]
        self.last_obs = [None for _ in range(num_envs)]
        self.mb_advs = None

        # Compute the shape of the input observation space, which may include
        # the contextual term.
        ob_dim = self._get_ob_dim(ob_space, co_space)

        # =================================================================== #
        # Step 1: Create input variables.                                     #
        # =================================================================== #

        with tf.compat.v1.variable_scope("input", reuse=False):
            self.rew_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name='rewards')
            self.action_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ac_space.shape,
                name='actions')
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,) + ob_dim,
                name='obs0')
            self.ret_ph = tf.placeholder(
                dtype=tf.float32,
                shape=(None,),
                name="ret_ph")
            self.advs_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="advs_ph")
            self.old_neglog_pac_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="old_neglog_pac_ph")
            self.old_vpred_ph = tf.compat.v1.placeholder(
                tf.float32,
                shape=(None,),
                name="old_vpred_ph")
            self.flat_tangent = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name="flat_tan")
            self.phase_ph = tf.compat.v1.placeholder(
                tf.bool,
                name='phase')
            self.rate_ph = tf.compat.v1.placeholder(
                tf.float32,
                name='rate')

        # =================================================================== #
        # Step 2: Create actor and critic variables.                          #
        # =================================================================== #

        # Create networks and core TF parts that are shared across setup parts.
        with tf.variable_scope("model", reuse=False):
            # Create the policy.
            self.action, self.pi_mean, self.pi_logstd = self.make_actor(
                self.obs_ph, scope="pi")
            self.pi_std = tf.exp(self.pi_logstd)

            # Create a method the log-probability of current actions.
            self.neglogp = self._neglogp(self.action)

            # Create the value function.
            self.value_fn = self.make_critic(self.obs_ph, scope="vf")
            self.value_flat = self.value_fn[:, 0]

        # Network for old policy
        with tf.variable_scope("oldpi/model", reuse=False):
            # Create the policy.
            self.old_action, self.old_pi_mean, self.old_pi_logstd = \
                self.make_actor(self.obs_ph, scope="pi")
            self.old_pi_std = tf.exp(self.old_pi_logstd)

            # Create a method the log-probability of current actions.
            self.old_neglogp = self._neglogp(self.old_action)

            # Create the value function.
            self.old_value_fn = self.make_critic(self.obs_ph, scope="vf")
            self.old_value_flat = self.old_value_fn[:, 0]

        # =================================================================== #
        # Step 3: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_optimizers(scope)

        # =================================================================== #
        # Step 4: Setup the operations for computing model statistics.        #
        # =================================================================== #

        self._setup_stats(scope or "Model")

    def make_actor(self, obs, reuse=False, scope="pi"):
        """Create an actor tensor.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the actor
        """
        # Initial image pre-processing (for convolutional policies).
        if self.model_params["model_type"] == "conv":
            pi_h = create_conv(
                obs=obs,
                image_height=self.model_params["image_height"],
                image_width=self.model_params["image_width"],
                image_channels=self.model_params["image_channels"],
                ignore_flat_channels=self.model_params["ignore_flat_channels"],
                ignore_image=self.model_params["ignore_image"],
                filters=self.model_params["filters"],
                kernel_sizes=self.model_params["kernel_sizes"],
                strides=self.model_params["strides"],
                act_fun=self.model_params["act_fun"],
                layer_norm=self.model_params["layer_norm"],
                batch_norm=self.model_params["batch_norm"],
                phase=self.phase_ph,
                dropout=self.model_params["dropout"],
                rate=self.rate_ph,
                scope=scope,
                reuse=reuse,
            )
        else:
            pi_h = obs

        # Create the output mean.
        policy_mean = create_fcnet(
            obs=pi_h,
            layers=self.model_params["layers"],
            num_output=self.ac_space.shape[0],
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            batch_norm=self.model_params["batch_norm"],
            phase=self.phase_ph,
            dropout=self.model_params["dropout"],
            rate=self.rate_ph,
            scope=scope,
            reuse=reuse,
        )

        # Create the output log_std.
        log_std = tf.get_variable(
            name='logstd',
            shape=[1, self.ac_space.shape[0]],
            initializer=tf.zeros_initializer()
        )

        # Create a method to sample from the distribution.
        std = tf.exp(log_std)
        action = policy_mean + std * tf.random_normal(
            shape=tf.shape(policy_mean),
            dtype=tf.float32
        )

        return action, policy_mean, log_std

    def make_critic(self, obs, reuse=False, scope="qf"):
        """Create a critic tensor.

        Parameters
        ----------
        obs : tf.compat.v1.placeholder
            the input observation placeholder
        reuse : bool
            whether or not to reuse parameters
        scope : str
            the scope name of the actor

        Returns
        -------
        tf.Variable
            the output from the critic
        """
        # Initial image pre-processing (for convolutional policies).
        if self.model_params["model_type"] == "conv":
            vf_h = create_conv(
                obs=obs,
                image_height=self.model_params["image_height"],
                image_width=self.model_params["image_width"],
                image_channels=self.model_params["image_channels"],
                ignore_flat_channels=self.model_params["ignore_flat_channels"],
                ignore_image=self.model_params["ignore_image"],
                filters=self.model_params["filters"],
                kernel_sizes=self.model_params["kernel_sizes"],
                strides=self.model_params["strides"],
                act_fun=self.model_params["act_fun"],
                layer_norm=self.model_params["layer_norm"],
                batch_norm=self.model_params["batch_norm"],
                phase=self.phase_ph,
                dropout=self.model_params["dropout"],
                rate=self.rate_ph,
                scope=scope,
                reuse=reuse,
            )
        else:
            vf_h = obs

        return create_fcnet(
            obs=vf_h,
            layers=self.model_params["layers"],
            num_output=1,
            stochastic=False,
            act_fun=self.model_params["act_fun"],
            layer_norm=self.model_params["layer_norm"],
            batch_norm=self.model_params["batch_norm"],
            phase=self.phase_ph,
            dropout=self.model_params["dropout"],
            rate=self.rate_ph,
            scope=scope,
            reuse=reuse,
        )

    def _setup_optimizers(self, scope):
        """Create the actor and critic optimizers."""
        pass  # TODO

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        pass  # TODO

    def initialize(self):
        """See parent class."""
        pass

    def step(self, obs):
        action, value = self.sess.run(
            [self.action, self.value_flat], feed_dict={self.obs_ph: obs})
        return action, value

    def logp(self, x, old=False):
        if old:
            return - self._old_neglogp(x)
        else:
            return - self._neglogp(x)

    def _neglogp(self, x):
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.pi_mean) / self.pi_std), axis=-1) + 0.5 * \
            np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.pi_logstd, axis=-1)

    def _old_neglogp(self, x):
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.old_pi_mean) / self.old_pi_std), axis=-1) + 0.5 * \
            np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.old_pi_logstd, axis=-1)

    def kl(self):
        return tf.reduce_sum(
            self.pi_logstd - self.old_pi_logstd + (
                tf.square(self.old_pi_std) +
                tf.square(self.old_pi_mean - self.pi_mean))
            / (2.0 * tf.square(self.pi_std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(
            self.pi_logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.pi_mean + self.pi_std * tf.random_normal(
            tf.shape(self.pi_mean), dtype=self.pi_mean.dtype)
