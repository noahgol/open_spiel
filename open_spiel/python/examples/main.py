# Runs CFR on a game and prints exploitability
# Michael Han 2024

import itertools as it
from absl import app
# from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import cfr_new as cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy
import pyspiel

games = ["kuhn_poker"]
# games = ["turn_based_simultaneous_game(game=goofspiel(players=3,imp_info=True,num_cards=4,points_order=descending))"]
# games = ["leduc_poker"]
# games = ["pig(horizon=5)"]
# games = ["hearts(pass_cards=false))"]

def print_policy(policy):
  for state, probs in zip(it.chain(*policy.states_per_player),
                          policy.action_probability_array):
    print(f'{state:6}   p={probs}')

def main(_):
    for game_name in games:
        print("Game: {}".format(game_name))
        game = pyspiel.load_game(game_name)
        cfr_solver = cfr.CFRSolver(game)

        T = 500
        for i in range(T):
            cfr_solver.evaluate_and_update_policy()
            if i % 10 == 0:
                # conv1 = exploitability.nash_conv(game, cfr_solver.average_policy())
                # print(f'Old exploitability after step {i} is {conv1}')
                conv2 = exploitability.new_nash_conv(game, cfr_solver.policy_history(), cfr_solver.average_policy(), None)
                print(f'New exploitability after step {i} is {conv2}')
                conv3 = exploitability.new_nash_conv(game, cfr_solver.policy_history(), cfr_solver.average_policy(), cfr_solver.policy_cache())
                print(f'Optimized exploitability after step {i} is {conv3}')

        # average_pol = cfr_solver.average_policy()
        # print_policy(average_pol)

if __name__ == "__main__":
    app.run(main)