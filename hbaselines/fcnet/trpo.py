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
from stable_baselines.common.mpi_adam import MpiAdam


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


def zipsame(*seqs):
    """
    Performs a zip function, but asserts that all zipped elements are of the
    same size

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
                self.sess,
                self.observation_space,
                self.action_space,
                reuse=False,
                **self.policy_kwargs,
            )

            # Network for old policy
            with tf.variable_scope("oldpi", reuse=False):
                self.old_policy = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    reuse=False,
                    **self.policy_kwargs,
                )

            with tf.variable_scope("loss", reuse=False):
                # Target advantage function (if applicable)
                self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])
                # Empirical return
                self.ret = tf.placeholder(dtype=tf.float32, shape=[None])

                observation = self.policy_tf.obs_ph
                self.action = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.action_space.shape[0]],
                    name="action_ph")

                kloldnew = self.old_policy.kl(self.policy_tf)
                ent = self.policy_tf.entropy()
                meankl = tf.reduce_mean(kloldnew)
                meanent = tf.reduce_mean(ent)
                entbonus = self.entcoeff * meanent

                vferr = tf.reduce_mean(
                    tf.square(self.policy_tf.value_flat - self.ret))

                # advantage * pnew / pold
                ratio = tf.exp(
                    self.policy_tf.logp(self.action) -
                    self.old_policy.logp(self.action))
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
            for (mbob, mbret) in iterbatches(
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


# =========================================================================== #
#                                    Policy                                   #
# =========================================================================== #

"""TRPO-compatible feedforward policy."""

from itertools import zip_longest


def mlp_extractor(flat_observations, net_arch, act_fun):
    latent = flat_observations
    # Layer sizes of the network that only belongs to the policy network
    policy_only_layers = []
    # Layer sizes of the network that only belongs to the value network
    value_only_layers = []

    # Iterate through the shared layers and build the shared parts of the
    # network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(
                latent, "shared_fc{}".format(idx), layer_size,
                init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), \
                "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), \
                    "Error: net_arch[-1]['pi'] must contain a list of " \
                    "integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), \
                    "Error: net_arch[-1]['vf'] must contain a list of " \
                    "integers."
                value_only_layers = layer['vf']
            # From here on the network splits up in policy and value network
            break

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(
            zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), \
                "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(
                latent_policy, "pi_fc{}".format(idx), pi_layer_size,
                init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), \
                "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(
                latent_value, "vf_fc{}".format(idx), vf_layer_size,
                init_scale=np.sqrt(2)))

    return latent_policy, latent_value


def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.

        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers,
        this corresponds to the fan-in, so this makes the initialization usable
        for both dense and convolutional layers.

        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        # pick the one with the correct shape
        weights = u if u.shape == flat_shape else v
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden],
                                 initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden],
                               initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias


class FeedForwardPolicy(object):
    """
    Policy object that implements actor critic, using a feed forward neural
    network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the
        Neural network for the policy (if None, default to [64, 64])
    :param act_fun: (tf.func) the activation function to use in the neural
        network.
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 reuse=False,
                 layers=None,
                 act_fun=tf.tanh):
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

        with tf.variable_scope("input", reuse=False):
            self.obs_ph = tf.placeholder(
                shape=(None,) + ob_space.shape,
                dtype=tf.float32,
                name="obs_ph")
            self.action_ph = None

        if layers is None:
            layers = [64, 64]
        net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(
                tf.layers.flatten(self.obs_ph), net_arch, act_fun)

            self.value_fn = linear(vf_latent, 'vf', 1)

            mean = linear(
                pi_latent,
                'pi',
                ac_space.shape[0],
                init_scale=0.01,
                init_bias=0.0)
            logstd = tf.get_variable(
                name='pi/logstd',
                shape=[1, ac_space.shape[0]],
                initializer=tf.zeros_initializer())

            self.policy = mean
            self.mean = mean
            self.logstd = logstd
            self.std = tf.exp(logstd)

        with tf.variable_scope("output", reuse=True):
            self.action = self.sample()
            self.deterministic_action = self.mean
            self.neglogp = self._neglogp(self.action)
            self.policy_proba = [self.mean, self.std]
            self.value_flat = self.value_fn[:, 0]

    def step(self, obs):
        action, value = self.sess.run(
            [self.action, self.value_flat], feed_dict={self.obs_ph: obs})
        return action, value

    def logp(self, x):
        return - self._neglogp(x)

    def _neglogp(self, x):
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.mean) / self.std), axis=-1) + 0.5 * \
            np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        return tf.reduce_sum(
            other.logstd - self.logstd +
            (tf.square(self.std) + tf.square(self.mean - other.mean)) /
            (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(
            self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(
            tf.shape(self.mean), dtype=self.mean.dtype)
