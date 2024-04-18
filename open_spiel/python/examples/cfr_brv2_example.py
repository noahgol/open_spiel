
"""Example use of the CFRBRv2 algorithm on Kuhn Poker."""

from absl import app
from absl import flags

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import cfr_br
from open_spiel.python.algorithms import cfr_brv2
from open_spiel.python.algorithms import exploitability_v2
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 100, "Number of iterations")
flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 10, "How often to print the exploitability")


def main(_):
  game = pyspiel.load_game(FLAGS.game, {"players": FLAGS.players})
  cfr_solver = cfr_brv2.CFRBRV2Solver(game)

  for i in range(FLAGS.iterations):
    cfr_solver.evaluate_and_update_policy()
    if i % FLAGS.print_freq == 0:
      #print(cfr_solver.average_policy(True).to_dict())
      conv = exploitability_v2.nash_conv_v2(game, cfr_solver.average_policy_by_iterations(i))
      print("Iteration {} nash_conv_v2 {}".format(i, conv))


if __name__ == "__main__":
  app.run(main)
