import sys
import threading
import urllib.parse
import json

import websocket

"""
    初始化websocket连接
"""
def init_connection(url, on_message_callback=None):
    ws = websocket.WebSocketApp(
        url, 
        on_open=ClientThread.on_open, 
        on_message=ClientThread.on_message,
        on_close=ClientThread.on_closed, 
        on_error=ClientThread.on_error
    )
    # 异步监听返回结果
    client = ClientThread(ws=ws, on_message_callback=on_message_callback)
    client.start()
    return client


"""
    初始化websocket连接, 并附带相关参数
"""
def init_connection_with_params(url, params, on_message_callback=None):
    url_prams_builder = urllib.parse.urlencode(params)
    url = url + '?' + url_prams_builder
    return init_connection(url, on_message_callback)


"""
    发送text message
"""
def send_text_message(ws, message):
    try:
        ws.send(message)
        print("send text message: " + str(message))
    except Exception as e:
        print(f"发送文本消息失败: {e}")


"""
    发送binary message
"""
def send_binary_message(ws, message):
    try:
        ws.send(message, websocket.ABNF.OPCODE_BINARY)
        print("send binary message length: " + str(len(message)))
    except Exception as e:
        print(f"发送二进制消息失败: {e}")
        raise e


class ClientThread(threading.Thread):
    def __init__(self, ws, on_message_callback=None):
        threading.Thread.__init__(self)
        self.ws = ws
        self.on_message_callback = on_message_callback
        ws.is_connect = False
        ws.connection_error = None

    def run(self):
        try:
            self.ws.run_forever()
        except Exception as e:
            print(f"WebSocket运行错误: {e}")
            self.ws.connection_error = e

    def return_is_connect(self):
        return self.ws.is_connect and not self.ws.connection_error

    def on_message(ws, message):
        try:
            print("received message: " + str(message))
            
            # 如果有自定义回调函数，调用它
            if hasattr(ws, 'on_message_callback') and ws.on_message_callback:
                try:
                    ws.on_message_callback(message)
                except Exception as e:
                    print(f"回调函数执行错误: {e}")
            
            # 该判断方式仅用作demo展示, 生产环境请使用json解析
            if "\"errorCode\":\"0\"" not in str(message):
                print("API返回错误，退出程序")
                sys.exit()
                
        except Exception as e:
            print(f"处理消息时出错: {e}")

    def on_open(ws):
        print("connection open")
        ws.is_connect = True
        ws.connection_error = None

    def on_closed(ws, close_status_code, close_msg):
        if not close_status_code:
            close_status_code = 'None'
        if not close_msg:
            close_msg = 'None'
        print("connection closed, code: " + str(close_status_code) + ", reason: " + str(close_msg))
        ws.is_connect = False

    def on_error(ws, error):
        print(f"WebSocket错误: {error}")
        ws.connection_error = error
        ws.is_connect = False