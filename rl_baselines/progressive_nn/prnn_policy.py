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


def prog_mlp_extractor(flat_observations, net_arch, act_fun, dict_res_tensor_ph, n_col=0):
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

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
            if (n_col > 0):
                with tf.variable_scope("pi_res_{}".format(idx),reuse=tf.AUTO_REUSE):
                    print(latent_policy.name)
                    # and"train_model" in latent_policy.name):
                    res_pi_ph = dict_res_tensor_ph[latent_policy.name.split(":")[0]]
                    printGreen(res_pi_ph)
                    res_len = res_pi_ph.shape[1]
                    U = tf.get_variable(name="U{}".format(idx), shape=[res_len, pi_layer_size],
                                        initializer=ortho_init(np.sqrt(2)))
                    latent_policy += tf.matmul(res_pi_ph , U)

            latent_policy = act_fun(latent_policy)

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."

            latent_value = (linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

            if (n_col > 0):
                with tf.variable_scope("vf_res_{}".format(idx),reuse=tf.AUTO_REUSE):
                    res_vf_ph = dict_res_tensor_ph[latent_value.name.split(":")[0]]
                    res_len = res_vf_ph.shape[1]
                    U = tf.get_variable(name="U{}".format(idx), shape=[res_len, vf_layer_size],
                                        initializer=ortho_init(np.sqrt(2)))
                    latent_value += tf.matmul(res_vf_ph , U)
            latent_value = act_fun(latent_value)


    return latent_policy, latent_value

#model/pi_fc1/add:0

class ProgressiveFeedForwardPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, prev_policy=[],prev_sess=[],
                 reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(ProgressiveFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                           n_batch, reuse=reuse, scale=(feature_extraction == "cnn"))
        self.prev_policy = prev_policy
        self.prev_sess = prev_sess
        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)
        # if none, then use two networks for policy and the value function
        n_prev_col = len(prev_policy)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        # create the residual placeholder for the progressive insertion
        if(len(self.prev_policy)>0):
            self.residualPlaceholder(ob_space, net_arch, prev_policy)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:    # construction of the model, mlp
                pi_latent, vf_latent = prog_mlp_extractor(tf.layers.flatten(self.processed_obs),
                                                          net_arch, act_fun,
                                                          self.dict_res_tensor_ph,
                                                          n_prev_col)

            self.value_fn = linear(vf_latent, 'vf', 1)
            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.initial_state = None
        self._setup_init()

    def residualPlaceholder(self, ob_space, net_arch=None, prev_policy=[]):
        n_prev_col = len(prev_policy)
        self.dict_res_tensor_ph = {}
        self.res_pi_tensor_ph = []
        self.res_vf_tensor_ph = []

        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):
                pass
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']

                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break  # From here on the network splits up in policy and value network


        with tf.variable_scope("res_input", reuse=tf.AUTO_REUSE):
            for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
                if pi_layer_size is not None:
                    name = "res_pi_fc{}".format(idx)
                    key = "train_model/model/pi_fc{}/add".format(idx)
                    res_pi_ph = tf.placeholder(dtype=ob_space.dtype, name=name,
                                                                   shape=[None, pi_layer_size * n_prev_col])
                    self.dict_res_tensor_ph[key] = res_pi_ph
                    self.res_pi_tensor_ph.append(res_pi_ph)
                    key = "model/pi_fc{}/add".format(idx)
                    name = "act_pi_fc{}".format(idx)
                    self.dict_res_tensor_ph[key] = tf.placeholder(dtype=ob_space.dtype, name=name,
                                                                  shape=[None, pi_layer_size * n_prev_col])


                if vf_layer_size is not None:
                    name = "res_vf_fc{}".format(idx)
                    key = "train_model/model/vf_fc{}/add".format(idx)
                    res_vf_ph = tf.placeholder(dtype=ob_space.dtype, name=name,
                                                                   shape=[None, vf_layer_size * n_prev_col])

                    self.dict_res_tensor_ph[key] = res_vf_ph
                    self.res_vf_tensor_ph.append(res_vf_ph)
                    name = "act_vf_fc{}".format(idx)
                    key = "model/vf_fc{}/add".format(idx)
                    self.dict_res_tensor_ph[key] = tf.placeholder(dtype=ob_space.dtype, name=name,
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

    def prev_feed_dict(self, obs):
        if(self.prev_policy.__len__()==0):
            return {}
        dict_value ={
            "model/pi_fc0/add":[],
            "model/pi_fc1/add":[],
            "model/vf_fc0/add":[],
            "model/vf_fc1/add":[],
            # "model/vf/add":[],
            # "model/pi/add":[],
            # "model/q/add":[],
        }
        for sess in self.prev_sess:
            feed_dict = {sess.graph.get_operation_by_name("input/Ob").values(): obs}
            graph = sess.graph
            all_operations = graph.get_operations()

            for ops in all_operations:
                name = ops.name
                # if ( "loss" not in name and "model" in name and "train" not in name
                #     and "add" in name and "res" not in name and "_" in name ):

                if (name in dict_value.keys()):
                    tensor = tf.squeeze(sess.graph.get_operation_by_name(name).values())
                    dict_value[name].append(sess.run(tensor,feed_dict))
        feed_dict = {self.dict_res_tensor_ph[key]:dict_value[key] for key in dict_value.keys()}
        return feed_dict


    def step(self, obs, state=None, mask=None, deterministic=False):
        feed_dict = self.prev_feed_dict(obs)

        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs, **feed_dict})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs, **feed_dict})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        feed_dict = self.prev_feed_dict(obs)
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, **feed_dict})

    def value(self, obs, state=None, mask=None):
        feed_dict = self.prev_feed_dict(obs)
        return self.sess.run(self._value, {self.obs_ph: obs, **feed_dict})


class ProgressiveMlpPolicy(ProgressiveFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, prev_policy=[],prev_sess=[],
                 reuse=False, **_kwargs):
        super(ProgressiveMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                   prev_policy=prev_policy,
                                                   prev_sess=prev_sess,
                                                   reuse=reuse,
                                        feature_extraction="mlp", **_kwargs)




