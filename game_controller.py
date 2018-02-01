from flask import request, jsonify
import numpy as np


def new_game():
    print(request.json['iterations'])
    return jsonify(np.array([3, 4, 5]))


def position_to_dto(position)
