import time
import tensorflow as tf
import numpy as np
import os
import csv
import random
from collections import deque

from hbaselines.utils.tf_util import make_session
from hbaselines.utils.tf_util import get_globals_vars
from hbaselines.utils.tf_util import get_trainable_vars

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import dataset
from stable_baselines.common import ActorCriticRLModel
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.runners import traj_segment_generator


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def zipsame(*seqs):
    """
    Performs a zip function, but asserts that all zipped elements are of the same size

    :param seqs: a list of arrays that are zipped together
    :return: the zipped arguments
    """
    length = len(seqs[0])
    assert all(len(seq) == length for seq in seqs[1:])
    return zip(*seqs)


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


class TRPO(ActorCriticRLModel):
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
        tensorflow). If None (default), use random seed. Note that if you want
        completely deterministic results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
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
                 seed=None,
                 n_cpu_tf_sess=1):
        self.policy = policy
        self.env = env
        self.verbose = verbose
        # self._requires_vec_env = False
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.params = None
        self.seed = seed
        self.n_cpu_tf_sess = n_cpu_tf_sess

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
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_fvp = None
        self.vfadam = None
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
            self.trainable_vars = self.params = self.setup_model()

    def setup_model(self):
        np.set_printoptions(precision=3)

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            self.sess = make_session(
                num_cpu=self.n_cpu_tf_sess, graph=self.graph)

            # Construct network for new policy
            self.policy_tf = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                self.n_envs,
                1,
                None,
                reuse=False,
                **self.policy_kwargs,
            )

            # Network for old policy
            with tf.variable_scope("oldpi", reuse=False):
                self.old_policy = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    self.n_envs,
                    1,
                    None,
                    reuse=False,
                    **self.policy_kwargs,
                )

            with tf.variable_scope("loss", reuse=False):
                # Target advantage function (if applicable)
                self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])
                # Empirical return
                self.ret = tf.placeholder(dtype=tf.float32, shape=[None])

                observation = self.policy_tf.obs_ph
                self.action = self.policy_tf.pdtype.sample_placeholder([None])

                kloldnew = self.old_policy.proba_distribution.kl(
                    self.policy_tf.proba_distribution)
                ent = self.policy_tf.proba_distribution.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                entbonus = self.entcoeff * meanent

                vferr = tf.reduce_mean(
                    tf.square(self.policy_tf.value_flat - self.ret))

                # advantage * pnew / pold
                ratio = tf.exp(
                    self.policy_tf.proba_distribution.logp(self.action) -
                    self.old_policy.proba_distribution.logp(self.action))
                surrgain = tf.reduce_mean(ratio * self.atarg)

                optimgain = surrgain + entbonus
                self.losses = [optimgain, meankl, entbonus, surrgain, meanent]
                self.loss_names = ["optimgain", "meankl", "entloss",
                                   "surrgain", "entropy"]

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
                flat_tangent = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None],
                    name="flat_tan")
                shapes = [var.get_shape().as_list() for var in var_list]
                start = 0
                tangents = []
                for shape in shapes:
                    var_size = tf_util.intprod(shape)
                    tangents.append(tf.reshape(
                        flat_tangent[start: start + var_size], shape))
                    start += var_size
                gvp = tf.add_n(
                    [tf.reduce_sum(grad * tangent)
                     for (grad, tangent) in zipsame(klgrads, tangents)])
                # Fisher vector products
                fvp = tf_util.flatgrad(gvp, var_list)

                self.assign_old_eq_new = tf_util.function(
                    [],
                    [],
                    updates=[tf.assign(oldv, newv) for (oldv, newv) in
                             zipsame(get_globals_vars("oldpi"),
                                     get_globals_vars("model"))],
                )
                self.compute_fvp = tf_util.function(
                    [flat_tangent, observation, self.old_policy.obs_ph,
                     self.action, self.atarg], fvp)
                self.vf_grad = tf_util.flatgrad(vferr, vf_var_list)

                tf_util.initialize(sess=self.sess)

                th_init = self.get_flat()
                self.set_from_flat(th_init)

            with tf.variable_scope("Adam_mpi", reuse=False):
                self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                self.vfadam.sync()

            self.grad = tf_util.flatgrad(optimgain, var_list)

        return get_trainable_vars("model") + get_trainable_vars("oldpi")

    def learn(self,
              total_timesteps,
              callback=None,
              log_interval=100,
              tb_log_name="TRPO",
              reset_num_timesteps=True):

        with self.sess.as_default():
            seg_gen = traj_segment_generator(
                self.policy_tf,
                self.env,
                self.timesteps_per_batch,
                reward_giver=None,
                gail=False,
                callback=self._init_callback(callback),
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
            return self.compute_fvp(
                vec, *fvpargs, sess=self.sess) + self.cg_damping * vec

        mean_losses = None

        add_vtarg_and_adv(seg, self.gamma, self.lam)
        atarg, tdlamret = seg["adv"], seg["tdlamret"]

        # predicted value function before update
        vpredbefore = seg["vpred"]
        # standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / (atarg.std() + 1e-8)

        args = seg["observations"], seg["observations"], seg["actions"], atarg
        # Subsampling: see p40-42 of John Schulman thesis
        # http://joschu.net/docs/thesis.pdf
        fvpargs = [arr[::5] for arr in args]

        self.assign_old_eq_new(sess=self.sess)

        # run loss backprop with summary, and save the metadata (memory,
        # compute time, ...)
        grad, *lossbefore = self.sess.run(
            [self.grad] + self.losses,
            feed_dict={
                self.policy_tf.obs_ph: seg["observations"],
                self.old_policy.obs_ph: seg["observations"],
                self.action: seg["actions"],
                self.atarg: atarg,
                self.ret: tdlamret,
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
                        self.old_policy.obs_ph: seg["observations"],
                        self.action: seg["actions"],
                        self.atarg: atarg,
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
            # NOTE: for recurrent policies, use shuffle=False?
            for (mbob, mbret) in dataset.iterbatches(
                    (seg["observations"], seg["tdlamret"]),
                    include_final_partial_batch=False,
                    batch_size=128,
                    shuffle=True):
                grad = self.sess.run(
                    self.vf_grad,
                    feed_dict={
                        self.policy_tf.obs_ph: mbob,
                        self.old_policy.obs_ph: mbob,
                        self.action: seg["actions"],
                        self.ret: mbret,
                    }
                )
                self.vfadam.update(grad, self.vf_stepsize)

        return mean_losses, vpredbefore, tdlamret

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

    def _get_pretrain_placeholders(self):
        pass

    def save(self, save_path, cloudpickle=False):
        pass


# =========================================================================== #
#                                    Policy                                   #
# =========================================================================== #

"""TRPO-compatible feedforward policy."""

from stable_baselines.common.policies import mlp_extractor
from stable_baselines.common.policies import linear
from stable_baselines.common.policies import make_proba_dist_type
from stable_baselines.common.policies import observation_input
from gym.spaces import Discrete


class FeedForwardPolicy(object):
    """
    Policy object that implements actor critic, using a feed forward neural
    network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the
        Neural network for the policy (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network
        architecture (see mlp_extractor documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural
        network.
    """

    recurrent = False

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 n_env,
                 n_steps,
                 n_batch,
                 reuse=False,
                 layers=None,
                 net_arch=None,
                 act_fun=tf.tanh):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        with tf.variable_scope("input", reuse=False):
            self._obs_ph, self._processed_obs = observation_input(
                ob_space, n_batch, scale=False)
            self._action_ph = None
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

        self._pdtype = make_proba_dist_type(ac_space)
        self._policy = None
        self._proba_distribution = None
        self._value_fn = None
        self._action = None
        self._deterministic_action = None

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(
                tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(
                    pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp],
                {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp],
                {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution \
                is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            self._policy_proba = [self.proba_distribution.mean,
                                  self.proba_distribution.std]
            self._value_flat = self.value_fn[:, 0]

    @property
    def pdtype(self):
        """ProbabilityDistributionType: type of the distribution for stochastic
        actions."""
        return self._pdtype

    @property
    def policy(self):
        """tf.Tensor: policy output, e.g. logits."""
        return self._policy

    @property
    def proba_distribution(self):
        """ProbabilityDistribution: distribution of stochastic actions."""
        return self._proba_distribution

    @property
    def value_fn(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._value_fn

    @property
    def value_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._value_flat

    @property
    def action(self):
        """tf.Tensor: stochastic action, of shape (self.n_batch, ) +
        self.ac_space.shape."""
        return self._action

    @property
    def deterministic_action(self):
        """tf.Tensor: deterministic action, of shape (self.n_batch, ) +
        self.ac_space.shape."""
        return self._deterministic_action

    @property
    def neglogp(self):
        """tf.Tensor: negative log likelihood of the action sampled by
        self.action."""
        return self._neglogp

    @property
    def policy_proba(self):
        """tf.Tensor: parameters of the probability distribution. Depends on
        pdtype."""
        return self._policy_proba

    @property
    def is_discrete(self):
        """bool: is action space discrete."""
        return isinstance(self.ac_space, Discrete)

    @property
    def initial_state(self):
        """
        The initial state of the policy. For feedforward policies, None. For a
        recurrent policy,
        a NumPy array of shape (self.n_env, ) + state_shape.
        """
        assert not self.recurrent, "When using recurrent policies, you must " \
                                   "overwrite `initial_state()` method"
        return None

    @property
    def obs_ph(self):
        """tf.Tensor: placeholder for observations, shape (self.n_batch, )
        + self.ob_space.shape."""
        return self._obs_ph

    @property
    def processed_obs(self):
        """tf.Tensor: processed observations, shape (self.n_batch, ) +
        self.ob_space.shape.

        The form of processing depends on the type of the observation space,
        and the parameters
        whether scale is passed to the constructor; see observation_input for
        more information."""
        return self._processed_obs

    @property
    def action_ph(self):
        """tf.Tensor: placeholder for actions, shape (self.n_batch, ) +
        self.ac_space.shape."""
        return self._action_ph
