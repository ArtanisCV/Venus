__author__ = 'Artanis'

import sys
import thread

import websocket


def on_message(ws, message):
    print message


def on_close(ws):
    print "Close connection from server."


def on_open(ws):
    def run(*args):
        while True:
            str = sys.stdin.readline().rstrip()
            ws.send(u"%s" % str)
        ws.close()

    thread.start_new_thread(run, ())


if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://localhost:8000/file",
                                on_open=on_open,
                                on_message=on_message,
                                on_close=on_close)
    ws.run_forever()