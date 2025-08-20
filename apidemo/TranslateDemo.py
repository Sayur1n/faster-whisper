import requests
import os
from pathlib import Path
from dotenv import load_dotenv

from apidemo.utils.AuthV3Util import addAuthParams

# 加载根目录的.env文件
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def createRequest(text, lang_from , lang_to, vocab_id = None):
    '''
    note: 将下列变量替换为需要请求的参数
    '''
    # 从环境变量加载API密钥
    APP_KEY = os.getenv('APP_KEY')
    APP_SECRET = os.getenv('APP_SECRET')
    
    if not APP_KEY or not APP_SECRET:
        raise ValueError("请在根目录的.env文件中设置APP_KEY和APP_SECRET")
    
    q = text 

    data = {'q': q, 'from': lang_from, 'to': lang_to, 'vocabId': vocab_id} if vocab_id else {'q': q, 'from': lang_from, 'to': lang_to}

    addAuthParams(APP_KEY, APP_SECRET, data)

    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    res = doCall('https://openapi.youdao.com/api', header, data, 'post')
    res = res.json()
    print(f"原文:{res['query']}")
    print(f"结果:{res['translation']}")
    return res 


def doCall(url, header, params, method):
    if 'get' == method:
        return requests.get(url, params)
    elif 'post' == method:
        return requests.post(url, params, header)

# 网易有道智云翻译服务api调用demo
# api接口: https://openapi.youdao.com/api
if __name__ == '__main__':
    createRequest("うみちゃんがこれから、書いたのって何日も、前だよな、なんか、まるで見てきたみたいだ", 'ja', 'zh-CHS')
