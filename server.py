from flask import Flask
import game_controller as game_c


app = Flask(__name__)

app.add_url_rule("/game", "game", game_c.new_game, methods=['POST'])


if __name__ == '__main__':
    app.run(debug=True, port=5001)
