from colosus.config import PlayerConfig
from colosus.game.position import Position
from colosus.game.square import Square
from colosus.player import Player


def prompt_bool(text, default: bool=True):
    options = "[y]/n"
    if not default:
        options = "y/[n]"

    value = input(f"{text} {options}: ")
    if len(value) == 0:
        return default

    return True if value.lower() != 'n' else False


def prompt_int(text, default: int):
    value = input(f"{text} [{default}]: ")
    if len(value) == 0:
        return default

    try:
        value = int(value)
        return value
    except ValueError:
        print(f"{text} must be an integer")
        return prompt_int(text, default)


def prompt_move():
    move = input(f"move (rank file): \n")
    move_parts = move.split(" ")
    try:
        rank = int(move_parts[0].strip())
        file = int(move_parts[1].strip())
        if not(0 <= rank <= 15 and 0 <= file <= 15):
            print(f"rank anf file must be an integers from 0 to 15")
            return prompt_move()
        else:
            return Square.square(rank, file)
    except ValueError:
        print(f"rank anf file must be an integers from 0 to 15")
        return prompt_move()


def prompt_legal_move(pos: Position):
    move = prompt_move()
    rank, file = Square.to_rank_file(move)
    if not pos.is_legal_colosus(move):
        print(f"{rank} {file} is not a legal move")
        return prompt_legal_move(pos)
    else:
        return move


def sort_policy(policy):
    move_policy = []
    for m in range(len(policy)):
        m_str = Square.to_string(m)
        move_policy.append((m_str, policy[m]))
    return sorted(move_policy, key=lambda t: t[1], reverse=True)


def print_policy_best_moves(policy, moves):
    sorted_policy = sort_policy(policy)
    for i in range(moves):
        m = sorted_policy[i]
        print("{} - {}".format(m[0], m[1]))


def print_state_children_stats(state):
    if state.children() is not None:
        for m in range(len(state.children())):
            c = state.children()[m]
            if c is not None:
                print("{} N: {}, W: {:.3g}, Q: {:.3g}, p:{:.3g}".format(Square.to_string(m), c.N, c.W, c.Q, c.P))


move_first = prompt_bool("Move first", True)
iterations = prompt_int("Iterations per move", 256)
show_colosus_data = prompt_bool("Show colosus data", True)
show_extra_colosus_data = prompt_bool("Show extra colosus data", False)
weights_filename = "./colosus/tests/c_10_500_1600.h5"

position = Position()
player = Player(PlayerConfig())
player.new_game(position, iterations, weights_filename)

person_turn = move_first
end = False

if person_turn:
    position.print()

while not end:
    if person_turn:
        move = prompt_legal_move(position)
        position = position.move(move)
        position.print()
        if position.is_end:
            print("You won!")
            break
        player.opponent_move(move)
    else:
        policy, value, move, old_state, new_state = player.move()
        if show_colosus_data:
            print(f"value: {value:.3}")
            print_policy_best_moves(policy, 5)
        if show_extra_colosus_data:
            print_state_children_stats(old_state)
        print(f"move: {Square.to_string(move)}")
        position = new_state.position()
        position.print()
        if position.is_end:
            print("You lost")
            break
    person_turn = not person_turn











