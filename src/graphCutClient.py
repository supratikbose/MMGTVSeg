#!/usr/bin/env python3
#This top line is really important - but why????

########################### Original, working ###################################
#https://realpython.com/python-sockets/
#https://github.com/realpython/materials/tree/master/python-sockets-tutorial

import os
import sys
import socket
import selectors
import traceback

#sys.path.append('/home/user/DMML/CodeAndRepositories/MMGTVSeg')
currentDirectory = os.path.dirname(os.path.abspath(__file__))
parentDirectory = os.path.join(currentDirectory, "..")
sys.path.append(parentDirectory)
import src
from src import libclient
#import libclient

#Moving it inside sendImCutRqstAndReceiveResult so that it gets called
# every time  else it was getting closed due to sel.close()
#sel = selectors.DefaultSelector()


# def create_request(action, value):
#     if action == "search":
#         return dict(
#             type="text/json",
#             encoding="utf-8",
#             content=dict(action=action, value=value),
#         )
#     else:
#         return dict(
#             type="binary/custom-client-binary-type",
#             encoding="binary",
#             content=bytes(action + value, encoding="utf-8"),
#         )

def create_request(action, value):
    if action == "imcut":
        request =  dict(
            type="text/json",
            encoding="utf-8",
            content=dict(action=action, value=value),
        )
        rqstFlag = True
    else:
        request =  dict()
        rqstFlag = False
    return rqstFlag, request

#def start_connection(host, port, request):
def start_connection(host, port, request, sel):    
    addr = (host, port)
    print("starting connection to", addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = libclient.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)

#def sendImCutRqstAndReceiveResult(host, port, graphCutInputConfig_JsonFilePath):
def sendImCutRqstAndReceiveResult(graphCutInputConfig_JsonFilePath):
    # if len(sys.argv) != 5:
    #     print("usage:", sys.argv[0], "<host> <port> <action> <value>")
    #     sys.exit(1)

    # host, port = sys.argv[1], int(sys.argv[2])
    # action, value = sys.argv[3], sys.argv[4]
    host, port = '127.0.0.1', 65432
    action, value = "imcut", graphCutInputConfig_JsonFilePath    
    #request = create_request(action, value)
    rqstFlag, request = create_request(action, value)
    if False == rqstFlag:
        print('Action: ', action, ' is not supported.')
    else:
        print('Starting action : ', action)

        #start_connection(host, port, request)
        sel = selectors.DefaultSelector()
        start_connection(host, port, request,sel)
        try:
            while True:
                events = sel.select(timeout=1)
                for key, mask in events:
                    message = key.data
                    try:
                        message.process_events(mask)
                        resultAvailable, result = message.getReply()
                    except Exception:
                        print(
                            "main: error: exception for",
                            f"{message.addr}:\n{traceback.format_exc()}",
                        )
                        message.close()
                # Check for a socket being monitored to continue.
                if not sel.get_map():
                    break
        except KeyboardInterrupt:
            print("caught keyboard interrupt, exiting")
        finally:
            sel.close()
    return resultAvailable, result

# ######### Test code ###########
# print('Trying.')
# resultAvailable, result = \
#     sendImCutRqstAndReceiveResult('/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment/graphCutInputConfig.json' )
# print('Got it.')
# print(result)

# print('Trying again.')
# resultAvailable, result = \
#     sendImCutRqstAndReceiveResult('/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment/graphCutInputConfig.json' )
# print('Got it.')
# print(result)

# print('Trying again.')
# resultAvailable, result = \
#     sendImCutRqstAndReceiveResult('/home/user/DMML/Data/PlayDataManualSegmentation/AutoScribbleExperiment/graphCutInputConfig.json' )
# print('Got it.')
# print(result)

# pass



