
import tornado.web
import tornado.gen

from mysql import MysqlHelper

from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from api2tx import test_raw_xgb,predict_raw_xgb
from api_l2tx_2 import test_raw
from es import is_indexExits,query


class PredictHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(32)

    config_json = json.load(open('config.json', 'r', encoding='utf-8'))
    mysql2_config = config_json['mysql2']

    host = mysql2_config.get('host')
    user = mysql2_config.get('user')
    password = mysql2_config.get('password')
    database_name = mysql2_config.get('database')
    port = mysql2_config.get('port')
    db = MysqlHelper(host, user, password, database_name, port=port)
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")  # 这个地方可以写域名
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Content-Type' , 'application/json;charset=UTF-8')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')

    ## 查询数据库的任务
    @tornado.gen.coroutine
    def get(self):
        module_name = self.get_argument('module_name','AAA')
        date = datetime.now().strftime('%Y.%m.%d')
        f=query(index='vkit123-' + date , startTime='now-5m', endTime='now',
              module_name=module_name)
        if len(f)!=0 :
            t2=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t1=(datetime.datetime.now() - datetime.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")
            result = predict_raw_xgb(f,t1,t2,module_name)
        self.write(result)

    # 处理OPTIONS请求
    @tornado.gen.coroutine
    def options(self):
        pass