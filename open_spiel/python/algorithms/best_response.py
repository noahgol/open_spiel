import collections
import itertools

import numpy as np

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python import policy as openspiel_policy
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_utils
import pyspiel

class BestResponsePolicy:
    def __init__(self, game, player_id, policy_history):
        self.game = game
        self.player_id = player_id
        self.policy_history = policy_history
        self.best_response_values = {}
        self.best_response_actions = {}
        self.game_tree = {}
        self.initialize_game_tree()
        self.compute_best_response()

    def initialize_game_tree(self):
        root_state = self.game.new_initial_state()
        if not root_state:
          raise ValueError("Failed to generate a valid initial state from the game.")
        self.process_state(root_state, [])

    def process_state(self, state, history):
        print(f"Processing state with history {history}")
        if state.is_terminal():
            self.game_tree[tuple(history)] = state.returns()[self.player_id]
            print(f"Adding terminal path: {history} with value: {state.returns()[self.player_id]}")
            return
        infostate = state.information_state_string(self.player_id)
        history.append(infostate)
        current_policy = self.policy_history[min(len(self.policy_history) - 1, len(history) - 1)]
        action_probs = current_policy.action_probabilities(state, self.player_id)
        print(f"Action probabilities for {infostate}: {action_probs}")

        legal_actions = state.legal_actions(self.player_id)
        print(f"Legal actions from {infostate}: {legal_actions}")
        for action, prob in action_probs.items():
            if prob > 0 and action in legal_actions:
                next_state = state.child(action)
                if next_state is None:
                    print(f"Failed to generate next state from action {action}")
                else:
                    self.process_state(next_state, history.copy())


    def compute_best_response(self):
        # Order paths by their length to start from the deepest nodes
        sorted_paths = sorted(self.game_tree.keys(), key=len, reverse=True)
        print('Sorted paths:', sorted_paths)
        for path in sorted_paths:
            print(f"Processing path: {path}")
            if isinstance(self.game_tree[path], dict):
              if isinstance(self.game_tree[path], (float, int)):  # Terminal node
                  self.best_response_values[path] = self.game_tree[path]
              else:
                  best_value = -float('inf')
                  best_action = None
                  action_values = {}

                  # Ensure that the game tree path correctly points to valid sub-dictionaries
                  if not isinstance(self.game_tree[path], dict):
                      continue  # Skip incorrectly formatted entries

                  for action, transitions in self.game_tree[path].items():
                        expected_value = 0
                        print(f"  Evaluating action: {action}")

                        # Iterate through possible transitions for this action
                        for next_state, prob in transitions.items():
                            next_path = path + (next_state,)
                            print(f"    Next path: {next_path}, Probability: {prob}")

                            # Check if the next path has a calculated best response value
                            if next_path in self.best_response_values:
                                expected_value += prob * self.best_response_values[next_path]
                                print(f"    Adding value from next path: {prob} * {self.best_response_values[next_path]}")
                            else:
                                print(f"    No best response value found for path {next_path}")

                        action_values[action] = expected_value
                        print(f"  Total expected value for action {action}: {expected_value}")

                        # Update best action and best value if this action is better
                        if expected_value > best_value:
                            best_value = expected_value
                            best_action = action
                            print(f"  New best action found: {action} with value {expected_value}")

                  # Store the best response and its value
                  self.best_response_values[path] = best_value
                  self.best_response_actions[path] = best_action
                  print(f"Storing best action for path {path}: {best_action}")


                  # Debugging output
                  print(f'Processed path {path}: Best Action = {best_action}, Best Value = {best_value}')
            else:
              print(f"Skipping non-dict path: {path}, value: {self.game_tree[path]}")
        # Debugging output to check final values
        print('Final best response values:', self.best_response_values)


    def value(self, state):
        infostate_tuple = tuple(state.information_state_string(self.player_id) for _ in range(len(self.policy_history)))
        return self.best_response_values.get(infostate_tuple, 0)

    def action_probabilities(self, state, player_id=None):
        infostate_tuple = tuple(state.information_state_string(self.player_id) for _ in range(len(self.policy_history)))
        best_action = self.best_response_actions.get(infostate_tuple)
        if best_action is None:
            print(f"No best action found for tuple {infostate_tuple}. Available actions: {state.legal_actions()}")
            return {action: 0 for action in state.legal_actions()}
        else:
            print(f"Best action found: {best_action} for tuple {infostate_tuple}")
            return {best_action: 1}