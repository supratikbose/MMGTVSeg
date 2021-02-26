#!/usr/bin/env python3
#This top line is really important - but why????

########################### Original, working ###################################
#https://realpython.com/python-sockets/
#https://github.com/realpython/materials/tree/master/python-sockets-tutorial

import sys
import socket
import selectors
import traceback
sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
import src
from src import libserver
#import libserver

sel = selectors.DefaultSelector()


def accept_wrapper(sock):
    conn, addr = sock.accept()  # Should be ready to read
    print("accepted connection from", addr)
    conn.setblocking(False)
    message = libserver.Message(sel, conn, addr)
    sel.register(conn, selectors.EVENT_READ, data=message)

def main():
    # if len(sys.argv) != 3:
    #     print("usage:", sys.argv[0], "<host> <port>")
    #     sys.exit(1)
    # host, port = sys.argv[1], int(sys.argv[2])
    host, port = '127.0.0.1', 65432 
    print('Starting graphcut server on local host and fixed port: ', port)
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Avoid bind() exception: OSError: [Errno 48] Address already in use
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((host, port))
    lsock.listen()
    print("listening on", (host, port))
    lsock.setblocking(False)
    sel.register(lsock, selectors.EVENT_READ, data=None)

    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_wrapper(key.fileobj)
                else:
                    message = key.data
                    try:
                        message.process_events(mask)
                    except Exception:
                        print(
                            "main: error: exception for",
                            f"{message.addr}:\n{traceback.format_exc()}",
                        )
                        message.close()
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()

if __name__ == "__main__":
    main()

