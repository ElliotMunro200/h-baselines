import time
import tensorflow as tf
import numpy as np
import os
import csv
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
        super(TRPO, self).__init__(
            policy=policy,
            env=env,
            verbose=verbose,
            requires_vec_env=False,
            _init_setup_model=_init_setup_model,
            policy_kwargs=policy_kwargs,
            seed=seed,
            n_cpu_tf_sess=n_cpu_tf_sess,
        )

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
            self.set_random_seed(self.seed)
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

        # setup_learn
        if self.episode_reward is None:
            self.episode_reward = np.zeros((self.n_envs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=100)

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
import numpy as np
import tensorflow as tf

from hbaselines.base_policies import Policy
from hbaselines.utils.tf_util import create_fcnet
from hbaselines.utils.tf_util import create_conv
from hbaselines.utils.tf_util import get_trainable_vars
from hbaselines.utils.tf_util import explained_variance
from hbaselines.utils.tf_util import print_params_shape
from hbaselines.utils.tf_util import process_minibatch


class FeedForwardPolicy(Policy):
    """Feed-forward neural network policy.

    Attributes
    ----------
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
        clipping parameter for the value function, it can be a function. This
        is a parameter specific to the OpenAI implementation. If None is passed
        (default), then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling. To deactivate
        value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    num_envs : int
        number of environments used to run simulations in parallel.
    mb_rewards : array_like
        a minibatch of environment rewards
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
    mb_dones : array_like
        a minibatch of done masks
    mb_all_obs : array_like
        a minibatch of full-state observations
    mb_returns : array_like
        a minibatch of expected discounted returns
    last_obs : array_like
        the most recent observation from each environment. Used to compute the
        GAE terms.
    mb_advs : array_like
        a minibatch of estimated advantages
    rew_ph : tf.compat.v1.placeholder
        placeholder for the rewards / discounted returns
    action_ph : tf.compat.v1.placeholder
        placeholder for the actions
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    advs_ph : tf.compat.v1.placeholder
        placeholder for the advantages
    old_neglog_pac_ph : tf.compat.v1.placeholder
        placeholder for the negative-log probability of the actions that were
        performed
    old_vpred_ph : tf.compat.v1.placeholder
        placeholder for the current predictions of the values of given states
    phase_ph : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    rate_ph : tf.compat.v1.placeholder
        the probability that each element is dropped if dropout is implemented
    action : tf.Variable
        the output from the policy/actor
    pi_mean : tf.Variable
        the output from the policy's mean term
    pi_logstd : tf.Variable
        the output from the policy's log-std term
    pi_std : tf.Variable
        the expnonential of the pi_logstd term
    neglogp : tf.Variable
        a differentiable form of the negative log-probability of actions by the
        current policy
    value_fn : tf.Variable
        the output from the value function
    value_flat : tf.Variable
        a one-dimensional (vector) version of value_fn
    entropy : tf.Variable
        computes the entropy of actions performed by the policy
    vf_loss : tf.Variable
        the output from the computed value function loss of a batch of data
    pg_loss : tf.Variable
        the output from the computed policy gradient loss of a batch of data
    approxkl : tf.Variable
        computes the KL-divergence between two models
    loss : tf.Variable
        the output from the computed loss of a batch of data
    optimizer : tf.Operation
        the operation that updates the trainable parameters of the actor
    """

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

        # =================================================================== #
        # Step 4: Setup the optimizers for the actor and critic.              #
        # =================================================================== #

        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.loss = None
        self.optimizer = None

        with tf.compat.v1.variable_scope("Optimizer", reuse=False):
            self._setup_optimizers(scope)

        # =================================================================== #
        # Step 5: Setup the operations for computing model statistics.        #
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

    def _neglogp(self, x):
        """Compute the negative log-probability of an input action (x)."""
        return 0.5 * tf.reduce_sum(
            tf.square((x - self.pi_mean) / self.pi_std), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) \
            * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.pi_logstd, axis=-1)

    def _setup_optimizers(self, scope):
        """Create the actor and critic optimizers."""
        pass  # TODO

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = {
            'reference_action_mean': tf.reduce_mean(self.pi_mean),
            'reference_action_std': tf.reduce_mean(self.pi_logstd),
            'rewards': tf.reduce_mean(self.rew_ph),
            'advantage': tf.reduce_mean(self.advs_ph),
            'old_neglog_action_probability': tf.reduce_mean(
                self.old_neglog_pac_ph),
            'old_value_pred': tf.reduce_mean(self.old_vpred_ph),
            'entropy_loss': self.entropy,
            'policy_gradient_loss': self.pg_loss,
            'value_function_loss': self.vf_loss,
            'approximate_kullback-leibler': self.approxkl,
            'clip_factor': self.clipfrac,
            'loss': self.loss,
            'explained_variance': explained_variance(
                self.old_vpred_ph, self.rew_ph)
        }

        tf.summary.scalar(
            'discounted_rewards', tf.reduce_mean(ret))
        tf.summary.scalar(
            'learning_rate', tf.reduce_mean(self.vf_stepsize))
        tf.summary.scalar(
            'advantage', tf.reduce_mean(atarg))
        tf.summary.scalar(
            'kl_clip_range', tf.reduce_mean(self.max_kl))

        tf.summary.scalar('entropy_loss', meanent)
        tf.summary.scalar('policy_gradient_loss', optimgain)
        tf.summary.scalar('value_function_loss', surrgain)
        tf.summary.scalar('approximate_kullback-leibler', meankl)
        tf.summary.scalar(
            'loss',
            optimgain + meankl + entbonus + surrgain + meanent)

        # Add all names and ops to the tensorboard summary.
        for key in ops.keys():
            name = "{}/{}".format(base, key)
            op = ops[key]
            tf.compat.v1.summary.scalar(name, op)

    def initialize(self):
        """See parent class."""
        pass

    def get_action(self, obs, context, apply_noise, random_actions, env_num=0):
        """See parent class."""
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        action, values, neglogpacs = self.sess.run(
            [self.action if apply_noise else self.pi_mean,
             self.value_flat, self.neglogp],
            feed_dict={
                self.obs_ph: obs,
                self.phase_ph: 0,
                self.rate_ph: 0.0,
            }
        )

        # Store information on the values and negative-log likelihood.
        self.mb_values[env_num].append(values)
        self.mb_neglogpacs[env_num].append(neglogpacs)

        return action

    def store_transition(self, obs0, context0, action, reward, obs1, context1,
                         done, is_final_step, env_num=0, evaluate=False):
        """Store a transition in the replay buffer.

        Parameters
        ----------
        obs0 : array_like
            the last observation
        context0 : array_like or None
            the last contextual term. Set to None if no context is provided by
            the environment.
        action : array_like
            the action
        reward : float
            the reward
        obs1 : array_like
            the current observation
        context1 : array_like or None
            the current contextual term. Set to None if no context is provided
            by the environment.
        done : float
            is the episode done
        is_final_step : bool
            whether the time horizon was met in the step corresponding to the
            current sample. This is used by the TD3 algorithm to augment the
            done mask.
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        evaluate : bool
            whether the sample is being provided by the evaluation environment.
            If so, the data is not stored in the replay buffer.
        """
        # Update the minibatch of samples.
        self.mb_rewards[env_num].append(reward)
        self.mb_obs[env_num].append(obs0.reshape(1, -1))
        self.mb_contexts[env_num].append(context0)
        self.mb_actions[env_num].append(action.reshape(1, -1))
        self.mb_dones[env_num].append(done)

        # Update the last observation (to compute the last value for the GAE
        # expected returns).
        self.last_obs[env_num] = self._get_obs([obs1], context1)

    def update(self, **kwargs):
        """See parent class."""
        # Compute the last estimated value.
        last_values = [
            self.sess.run(
                self.value_flat,
                feed_dict={
                    self.obs_ph: self.last_obs[env_num],
                    self.phase_ph: 0,
                    self.rate_ph: 0.0,
                })
            for env_num in range(self.num_envs)
        ]

        (self.mb_obs,
         self.mb_contexts,
         self.mb_actions,
         self.mb_values,
         self.mb_neglogpacs,
         self.mb_all_obs,
         self.mb_rewards,
         self.mb_returns,
         self.mb_dones,
         self.mb_advs, n_steps) = process_minibatch(
            mb_obs=self.mb_obs,
            mb_contexts=self.mb_contexts,
            mb_actions=self.mb_actions,
            mb_values=self.mb_values,
            mb_neglogpacs=self.mb_neglogpacs,
            mb_all_obs=self.mb_all_obs,
            mb_rewards=self.mb_rewards,
            mb_returns=self.mb_returns,
            mb_dones=self.mb_dones,
            last_values=last_values,
            gamma=self.gamma,
            lam=self.lam,
            num_envs=self.num_envs,
        )

        # Run the optimization procedure.
        batch_size = n_steps // self.n_minibatches

        inds = np.arange(n_steps)
        for _ in range(self.n_opt_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                mbinds = inds[start:end]
                self.update_from_batch(
                    obs=self.mb_obs[mbinds],
                    context=None if self.mb_contexts[0] is None
                    else self.mb_contexts[mbinds],
                    returns=self.mb_returns[mbinds],
                    actions=self.mb_actions[mbinds],
                    values=self.mb_values[mbinds],
                    advs=self.mb_advs[mbinds],
                    neglogpacs=self.mb_neglogpacs[mbinds],
                )

    def update_from_batch(self,
                          obs,
                          context,
                          returns,
                          actions,
                          values,
                          advs,
                          neglogpacs):
        """Perform gradient update step given a batch of data.

        Parameters
        ----------
        obs : array_like
            a minibatch of observations
        context : array_like
            a minibatch of contextual terms
        returns : array_like
            a minibatch of contextual expected discounted returns
        actions : array_like
            a minibatch of actions
        values : array_like
            a minibatch of estimated values by the policy
        advs : array_like
            a minibatch of estimated advantages
        neglogpacs : array_like
            a minibatch of the negative log-likelihood of performed actions
        """
        # Add the contextual observation, if applicable.
        obs = self._get_obs(obs, context, axis=1)

        return self.sess.run(self.optimizer, {
            self.obs_ph: obs,
            self.action_ph: actions,
            self.advs_ph: advs,
            self.rew_ph: returns,
            self.old_neglog_pac_ph: neglogpacs,
            self.old_vpred_ph: values,
            self.phase_ph: 1,
            self.rate_ph: 0.5,
        })

    def get_td_map(self):
        """See parent class."""
        # Add the contextual observation, if applicable.
        context = None if self.mb_contexts[0] is None else self.mb_contexts
        obs = self._get_obs(self.mb_obs, context, axis=1)

        td_map = self.get_td_map_from_batch(
            obs=obs.copy(),
            mb_actions=self.mb_actions,
            mb_advs=self.mb_advs,
            mb_returns=self.mb_returns,
            mb_neglogpacs=self.mb_neglogpacs,
            mb_values=self.mb_values,
        )

        # Clear memory
        self.mb_rewards = [[] for _ in range(self.num_envs)]
        self.mb_obs = [[] for _ in range(self.num_envs)]
        self.mb_contexts = [[] for _ in range(self.num_envs)]
        self.mb_actions = [[] for _ in range(self.num_envs)]
        self.mb_values = [[] for _ in range(self.num_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.num_envs)]
        self.mb_dones = [[] for _ in range(self.num_envs)]
        self.mb_all_obs = [[] for _ in range(self.num_envs)]
        self.mb_returns = [[] for _ in range(self.num_envs)]
        self.last_obs = [None for _ in range(self.num_envs)]
        self.mb_advs = None

        return td_map

    def get_td_map_from_batch(self,
                              obs,
                              mb_actions,
                              mb_advs,
                              mb_returns,
                              mb_neglogpacs,
                              mb_values):
        """Convert a batch to a td_map."""
        return {
            self.obs_ph: obs,
            self.action_ph: mb_actions,
            self.advs_ph: mb_advs,
            self.rew_ph: mb_returns,
            self.old_neglog_pac_ph: mb_neglogpacs,
            self.old_vpred_ph: mb_values,
            self.phase_ph: 0,
            self.rate_ph: 0.0,
        }
