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

"""Example use of the CFR algorithm on Kuhn Poker."""

from absl import app
from absl import flags

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 10, "Number of iterations")
flags.DEFINE_string("game", "pig", "Name of the game")
flags.DEFINE_integer("players", 3, "Number of players")
flags.DEFINE_integer("horizon", 5, "horizon")
flags.DEFINE_integer("print_freq", 1, "How often to print the exploitability")


def main(_):
#  game = pyspiel.load_game("pig", {"players": FLAGS.players, "horizon" : FLAGS.horizon})
  game = pyspiel.load_game("kuhn_poker", {"players": FLAGS.players})
#  game = pyspiel.load_game_as_turn_based("goofspiel", 
#      {"players": 3, "imp_info": True, "num_cards": 4, "points_order": "descending"})
  
  cfr_solver = cfr.CFRSolver(game)

  for i in range(FLAGS.iterations):
    cfr_solver.evaluate_and_update_policy()
    if (i+1) % FLAGS.print_freq == 0:
      conv = exploitability.nash_conv(game, cfr_solver.average_policy())
      nash_conv = exploitability.nash_conv(game, cfr_solver.average_policy(),
                                           policy_history=cfr_solver.policy_history())
      print("Iteration {0:4d} exploitability {1:.15f} new nash_conv {2:.15f}".format(i+1, conv, nash_conv))


if __name__ == "__main__":
  app.run(main)
