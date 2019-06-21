import pdb
from pdb import set_trace
import numpy as np
import tensorflow as tf
import sys
import multiprocessing


from itertools import zip_longest
import warnings


from stable_baselines.common import tf_util, SetVerbosity
from stable_baselines.common.policies import nature_cnn, ActorCriticPolicy, LstmPolicy
from stable_baselines.a2c.utils import ortho_init
from stable_baselines import PPO2


from srl_zoo.utils import printRed,printGreen,printYellow


def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    """
    Creates a fully connected layer for TensorFlow

    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
    :param scope: (str) The TensorFlow variable scope
    :param n_hidden: (int) The number of hidden neurons
    :param init_scale: (int) The initialization scale
    :param init_bias: (int) The initialization offset bias
    :return: (TensorFlow Tensor) fully connected layer
    """
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias


def prog_mlp_extractor(flat_observations, net_arch, act_fun, dict_res_tensor={}):
    printRed(net_arch)
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Test purpose
    #
    # if(len(prev_cols)>0):
    #     model = prev_cols[0]
    #     sess = model.sess
    #     graph =sess.graph
    #
    #     model_variables = graph.get_operations()
    #     train_ops = []
    #     for op in model_variables:
    #         if("loss" not in str(op.name) and "model" in str(op.name) and "add" in str(op.name)):
    #             print(op.name)
    #             train_ops.append(op.values())
    #             # try:
    #             #     tmp = sess.run(op.values())
    #             #     if(tmp !=None):
    #             #         print(op.name,op.values(), tmp[0].shape)
    #             # except:
    #             #     print("feed_dict no compitible", op.name)
    #     #model / pi_fc0 / w: 0
    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = (linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))
            print(latent_policy.name)
            with tf.variable_scope("pi_res_{}".format(idx)):
                try:
                    print(dict_res_tensor[latent_policy.name])
                except:
                    printRed("Exception: residual insertion from previous model failed.")
            latent_policy = act_fun(latent_policy)
        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = (linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))
            with tf.variable_scope("vf_res_{}".format(idx)):
                latent_value = act_fun(latent_value + 2)
    return latent_policy, latent_value

#model/pi_fc1/add:0

class ProgressiveFeedForwardPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_prev_col=0, tensor_name=[],
                 reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(ProgressiveFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                           n_batch, reuse=reuse, scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)
        

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)
        # if none, then use two networks for policy and the value function



        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        # create the residual placeholder for the progressive insertion
        self._residualPlaceholder(ob_space,net_arch, n_prev_col)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:    # construction of the model, mlp
                pi_latent, vf_latent = prog_mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun, n_prev_col)

            self.value_fn = linear(vf_latent, 'vf', 1)
            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.initial_state = None
        self._setup_init()

    def _residualPlaceholder(self, ob_space, net_arch=None, n_prev_col=0):
        self.dict_res_tensor_ph = {}
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):
                pass;
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']

                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break  # From here on the network splits up in policy and value network


        with tf.variable_scope("res_input", reuse=False):
            for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
                if pi_layer_size is not None:
                    name = "res_pi_fc{}".format(idx)
                    self.dict_res_tensor_ph[name] = tf.placeholder(dtype=ob_space.dtype, name=name,
                                                                   shape=[None, pi_layer_size * n_prev_col])
                if vf_layer_size is not None:
                    name = "res_vf_fc{}".format(idx)
                    self.dict_res_tensor_ph[name] = tf.placeholder(dtype=ob_space.dtype, name=name,
                                                                   shape=[None, vf_layer_size * n_prev_col])



    @staticmethod
    def _kwargs_check(feature_extraction, kwargs):
        """
        Ensure that the user is not passing wrong keywords
        when using policy_kwargs.

        :param feature_extraction: (str)
        :param kwargs: (dict)
        """
        # When using policy_kwargs parameter on model creation,
        # all keywords arguments must be consumed by the policy constructor except
        # the ones for the cnn_extractor network (cf nature_cnn()), where the keywords arguments
        # are not passed explicitly
        # (using **kwargs to forward the arguments)
        # that's why there should be not kwargs left when using the mlp_extractor
        # (in that case the keywords arguments are passed explicitely)
        if feature_extraction == 'mlp' and len(kwargs) > 1:
            raise ValueError("Unknown keywords for policy: {}".format(kwargs))


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


class ProgressiveMlpPolicy(ProgressiveFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_prev_col=0, tensor_name=[], reuse=False, **_kwargs):
        super(ProgressiveMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                   n_prev_col=n_prev_col, tensor_name=tensor_name, reuse=reuse,
                                        feature_extraction="mlp", **_kwargs)










class ProgPPO2(PPO2):
    def __init__(self, policy, env, prev_cols=(),gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False):
        """
        Parameters could be referred to those of PPO2 in stable_baselines
        :param prev_cols: The previous pre-trained model, the progressive model
        """
        super(ProgPPO2, self).__init__(
            policy=policy, env=env, verbose=verbose, _init_setup_model=False, policy_kwargs=policy_kwargs)
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.prev_cols = prev_cols

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.dict_res_tensor_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        if _init_setup_model:
            self.setup_progressive_model()

    def setup_progressive_model(self):

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                n_batch_step = None
                n_batch_train = None

                # TODO: update the prnn for lstm policy
                if issubclass(self.policy, LstmPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                tensor_name, _ = self.getResidualTensorName()
                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, n_prev_col=len(self.prev_cols), tensor_name=tensor_name,
                                        reuse=False, **self.policy_kwargs)

                with tf.variable_scope("train_model", reuse=True, custom_getter=tf_util.outer_scope_getter("train_model")):

                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              n_prev_col=len(self.prev_cols), tensor_name=tensor_name,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    #action placeholder
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    #advantages placeholder
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model._value
                    vpredclipped = self.old_vpred_ph + tf.clip_by_value(
                        train_model._value - self.old_vpred_ph, - self.clip_range_ph, self.clip_range_ph)
                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpredclipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leiber', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()


    def getResidualTensorName(self):
        res_tensor_name = []
        res_tensor_value = {}
        if(len(self.prev_cols)==0):
            return "", ""
        graph = self.prev_cols[0].sess.graph
        all_operations = graph.get_operations()
        for ops in all_operations:
            name = ops.name
            if (("loss" not in name) and ("train_model/model/" in name) and ("add" in name) and "res" not in name):
                res_tensor_name.append(name)
                res_tensor_value[name] = []
        return res_tensor_name, res_tensor_value

    """
    This part is very slow, should get quiker.
    """
    def residualBlock(self, obs):
        res_tensor_name, res_tensor_value = self.getResidualTensorName()

        for model in self.prev_cols:
            sess = model.sess
            prev_graph = sess.graph
            feed_dict = {prev_graph.get_operation_by_name('train_model/input/Ob').values(): obs}
            for name in res_tensor_name:
                tensor = prev_graph.get_operation_by_name(name).values()
                res_tensor_value[name].append((sess.run(tf.squeeze(tensor), feed_dict)))
        if(len(self.prev_cols)>0):
            for name in res_tensor_name:
                if(len(res_tensor_value[name][0].shape)==1):
                    batch_size = res_tensor_value[name][0].shape[0]
                    for i,arr in enumerate(res_tensor_value[name]):
                        res_tensor_value[name][i] = arr.reshape([batch_size,1])
                res_tensor_value[name] = np.concatenate(res_tensor_value[name],axis=1)

        return res_tensor_value, res_tensor_name

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        res_tensor_value, res_tensor_name = self.residualBlock(obs)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions, self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values, self.dict_res_tensor_ph:res_tensor_value}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.masks_ph] = masks

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1




        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)


        # ### study baselines
        # graph =self.sess.graph
        # model_variables =graph.get_operations()
        # train_ops =[]
        # for op in model_variables:
        #     if("loss" not in str(op.name) and "Tanh" in str(op.name)):
        #         #print(op.name)
        #         train_ops.append(op)
        #         try:
        #             tmp = self.sess.run(op.values(),td_map)
        #             if(tmp !=None):
        #                 print(op.name,op.values(), tmp[0].shape)
        #         except:
        #             print("feed_dict no compitible", op.name)
        # import pdb

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac