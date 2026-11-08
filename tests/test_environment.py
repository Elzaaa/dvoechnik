import chess

from dvoechnik.environment import ChessEnvironment


def test_initial_position_has_legal_actions():
    env = ChessEnvironment(seed=0)
    observation = env.reset()
    assert len(observation) == 769
    assert len(env.legal_actions()) == 20


def test_step_changes_position():
    env = ChessEnvironment(seed=0)
    before = env.observe()
    env.step(env.encode_action(chess.Move.from_uci("e2e4")))
    assert env.observe() != before
    assert env.board.fullmove_number == 1


def test_illegal_action_is_rejected():
    env = ChessEnvironment()
    env.reset()
    try:
        env.step(env.encode_action(chess.Move.from_uci("e2e5")))
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal action was accepted")
