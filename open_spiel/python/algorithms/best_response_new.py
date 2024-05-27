# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computes a Best-Response policy.

The goal if this file is to be the main entry-point for BR APIs in Python.

TODO(author2): Also include computation using the more efficient C++
`TabularBestResponse` implementation.
"""

import collections
import itertools
import pdb
import numpy as np

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import noisy_policy
from open_spiel.python.algorithms import policy_utils
import pyspiel


def _memoize_method(key_fn=lambda x: x):
  """Memoize a single-arg instance method using an on-object cache."""

  def memoizer(method):
    cache_name = "cache_" + method.__name__

    def wrap(self, arg):
      key = key_fn(arg)
      cache = vars(self).setdefault(cache_name, {})
      if key not in cache:
        cache[key] = method(self, arg)
      return cache[key]

    return wrap

  return memoizer

class BestResponsePolicy(openspiel_policy.Policy):
  """Computes the best response to a specified strategy."""

  def __init__(self,
               game,
               player_id,
               policy_history,
               policy_cache,
               root_state=None,
               cut_threshold=0.0):
    """Initializes the best-response calculation.

    Args:
      game: The game to analyze.
      player_id: The player id of the best-responder.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root state is used.
      cut_threshold: The probability to cut when calculating the value.
        Increasing this value will trade off accuracy for speed.
    """
    self._num_players = game.num_players()
    self._player_id = player_id
    self._policy_history = policy_history
    self._policy_cache = policy_cache
    if root_state is None:
      root_state = game.new_initial_state()
    self._root_state = root_state
    self.infosets = [self.info_sets(root_state, policy) for policy in policy_history]
    self._cut_threshold = cut_threshold
    if policy_cache:
      policy_cache[2][self._player_id] = len(policy_history)


  def info_sets(self, state, policy):
    """Returns a dict of infostatekey to list of (state, cf_probability)."""
    infosets = collections.defaultdict(list)
    for s, p in self.decision_nodes_per_policy(state, policy):
      infosets[s.information_state_string(self._player_id)].append((s, p))
    return dict(infosets)

  def transitions_per_policy(self, state, policy):
    """Returns a list of (action, cf_prob) pairs from the specified state."""
    if state.current_player() == self._player_id:
      # Counterfactual reach probabilities exclude the best-responder's actions,
      # hence return probability 1.0 for every action.
      return [(action, 1.0) for action in state.legal_actions()]
    elif state.is_chance_node():
      return state.chance_outcomes()
    elif state.is_simultaneous_node():
      return self.joint_action_probabilities_counterfactual(state)
    else:
      return list(policy.action_probabilities(state).items())

  def decision_nodes_per_policy(self, parent_state, policy):
    """Yields a (state, cf_prob) pair for each descendant decision node."""
    if not parent_state.is_terminal():
      if (parent_state.current_player() == self._player_id or   
          parent_state.is_simultaneous_node()):
        yield (parent_state, 1.0)
      for action, p_action in self.transitions_per_policy(parent_state, policy):
        for state, p_state in self.decision_nodes_per_policy(
            openspiel_policy.child(parent_state, action), policy):
          yield (state, p_state * p_action)

  def all_descendant_states(self, parent_state):
    if not parent_state.is_terminal():
      if (parent_state.current_player() == self._player_id or
          parent_state.is_simultaneous_node()):
        yield parent_state
      for action in parent_state.legal_actions():
        for child_state in self.all_descendant_states(parent_state.child(action)):
          yield child_state

  @_memoize_method(key_fn=lambda state: state.history_str())
  def decision_nodes_slow(self, parent_state):
    # for each t in [T], call old decision_nodes() with policy_history[t]
    # average reach probabilities for each state over t in [T].
    # return list of (state, average_reach_prob) pairs

    if not parent_state.is_terminal():
      if (parent_state.current_player() == self._player_id or
          parent_state.is_simultaneous_node()):
        yield (parent_state, 1.0)
      T = len(self._policy_history)
      prob_sum = {}
      for policy in self._policy_history:
        for state, p_state in self.decision_nodes_per_policy(parent_state, policy):
          hist = state.history_str()
          if hist not in prob_sum:
            prob_sum[hist] = [state, p_state]
          else:
            prob_sum[hist][1] += p_state
      for hist in prob_sum:
        state, p_state = prob_sum[hist]
        yield (state, p_state/T)

  @_memoize_method(key_fn=lambda state: state.history_str())
  def decision_nodes(self, parent_state):
    # for each t in [T], call old decision_nodes() with policy_history[t]
    # average reach probabilities for each state over t in [T].
    # return list of (state, average_reach_prob) pairs

    if not parent_state.is_terminal():
      if (parent_state.current_player() == self._player_id or
          parent_state.is_simultaneous_node()):
        yield (parent_state, 1.0)
      T = len(self._policy_history)
      cache_T = 0
      cache = self._policy_cache
      if cache:
        cache_T = cache[2][self._player_id]
      prob_sum = {}
      for policy in self._policy_history[cache_T:]:
        for state, p_state in self.decision_nodes_per_policy(parent_state, policy):
          hist = state.history_str()
          if cache:
            ground_state_node = cache[1][hist]
            ground_state_node.cumulative_p_state += p_state
          else:
            if hist not in prob_sum:
              prob_sum[hist] = [state, p_state]
            else:
              prob_sum[hist][1] += p_state
      if cache:
        for state in self.all_descendant_states(parent_state):
          hist = state.history_str()
          ground_state_node = cache[1][hist]
          yield (state, ground_state_node.cumulative_p_state/T)
      else:
        for hist in prob_sum:
          state, p_state = prob_sum[hist]
          yield (state, p_state/T)

  def joint_action_probabilities_counterfactual(self, state):
    """Get list of action, probability tuples for simultaneous node.

    Counterfactual reach probabilities exclude the best-responder's actions,
    the sum of the probabilities is equal to the number of actions of the
    player _player_id.
    Args:
      state: the current state of the game.

    Returns:
      list of action, probability tuples. An action is a tuple of individual
        actions for each player of the game.
    """
    actions_per_player, probs_per_player = (
        openspiel_policy.joint_action_probabilities_aux(state, self._policy))
    probs_per_player[self._player_id] = [
        1.0 for _ in probs_per_player[self._player_id]
    ]
    return [(list(actions), np.prod(probs)) for actions, probs in zip(
        itertools.product(
            *actions_per_player), itertools.product(*probs_per_player))]
    
  def avg_action_probs_old(self, state):
    policy_history = self._policy_history
    num_policies = len(policy_history)
    cumul = {action : 0.0 for action in policy_history[0].action_probabilities(state)}
    for policy in policy_history:
      for action, prob in policy.action_probabilities(state).items():
        cumul[action] += prob
    return {action : cumul[action]/num_policies for action in cumul}.items()

  def avg_action_probs(self, state):
    cache = self._policy_cache
    cache_T = 0
    if cache:
      hist = state.history_str()
      ground_state_node = cache[1][hist]
      cache_T = ground_state_node.transition_T
    else:
      cumul = {action : 0.0 for action in self._policy_history[0].action_probabilities(state)}
    T = len(self._policy_history)
    for policy in self._policy_history[cache_T:]:
      for action, prob in policy.action_probabilities(state).items():
        if cache:
          ground_state_node.cumulative_p_action[action] += prob
        else:
          cumul[action] += prob
    if cache:
      ground_state_node.transition_T = T
      return {action : ground_state_node.cumulative_p_action[action]/T for action in state.legal_actions()}.items()
    else:
      return {action : cumul[action]/T for action in cumul}.items()

  def transitions(self, state):
    """Returns a list of (action, cf_prob) pairs from the specified state."""
    if state.current_player() == self._player_id:
      return [(action, 1.0) for action in state.legal_actions()]
    elif state.is_chance_node():
      return state.chance_outcomes()
    else:
        return list(self.avg_action_probs(state))

  def value_root(self, state):
    return sum(self.value(state, self._policy_history[i]) for i in range(len(self._policy_history)))/len(self._policy_history)
      
  # @_memoize_method(key_fn=lambda state: state.history_str())
  def value(self, state, policy):
    """Returns the value of the specified state to the best-responder."""
    if state.is_terminal():
      return state.player_return(self._player_id)
    elif (state.current_player() == self._player_id or
          state.is_simultaneous_node()):
      action = self.best_response_action(
          state.information_state_string(self._player_id))
      return self.q_value(state, action, policy)
    else:
      return sum(p * self.q_value(state, a, policy)
                 for a, p in self.transitions_per_policy(state, policy)
                 if p > self._cut_threshold)

  def q_value(self, state, action, policy):
    """Returns the value of the (state, action) to the best-responder."""
    if state.is_simultaneous_node():

      def q_value_sim(sim_state, sim_actions):
        child = sim_state.clone()
        # change action of _player_id
        sim_actions[self._player_id] = action
        child.apply_actions(sim_actions)
        return self.value(child)

      actions, probabilities = zip(*self.transitions(state))
      return sum(p * q_value_sim(state, a)
                 for a, p in zip(actions, probabilities / sum(probabilities))
                 if p > self._cut_threshold)
    else:
      return self.value(state.child(action), policy)

  @_memoize_method()
  def best_response_action(self, infostate):
    """Returns the best response for this information state."""
    infoset = [self.infosets[i][infostate] for i in range(len(self._policy_history))]
    # Get actions from the first (state, cf_prob) pair in the infoset list.
    # Return the best action by counterfactual-reach-weighted state-value.
    def result(a):
      count = 0
      for i in range(len(self._policy_history)):
        count += sum(cf_p * self.q_value(s, a, self._policy_history[i]) for s, cf_p in infoset[i])
      return count
    return max(
        infoset[0][0][0].legal_actions(self._player_id),
        key=result)

  def action_probabilities(self, state, player_id=None):
    """Returns the policy for a player in a state.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
    if player_id is None:
      if state.is_simultaneous_node():
        player_id = self._player_id
      else:
        player_id = state.current_player()
    return {
        self.best_response_action(state.information_state_string(player_id)): 1
    }
