import asyncio
from concurrent.futures import ThreadPoolExecutor

import time


def caller():
    new_game(el_callback)


def el_callback(x):
    print("ejecuto callback")
    print(f"arg: {x}")


def new_game(callback):
    executor = ThreadPoolExecutor()
    future = executor.submit(long_task)
    future.add_done_callback(lambda f: callback(f.result()))
    print("termino new game")


def long_task():
    time.sleep(5)
    return "hola"


caller()
