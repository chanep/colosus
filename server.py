from flask import Flask
import match_controller as match_c


app = Flask(__name__)

app.add_url_rule("/match", "match", match_c.new_match, methods=['POST'])


if __name__ == '__main__':
    app.run(debug=True, port=5001)
