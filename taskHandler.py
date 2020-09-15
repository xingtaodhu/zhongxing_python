
import tornado.web
import tornado.gen
import _thread
from mysql import MysqlHelper
from sqledit import *
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
import json
from api2tx import test_raw_xgb,predict_raw_xgb
from api_l2tx_2 import test_raw
from es import is_indexExits,query
from utils import timestamp2iso,getEveryDay


class TaskHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(32)

    config_json = json.load(open('config.json', 'r', encoding='utf-8'))
    mysql2_config = config_json['mysql2']

    host = mysql2_config.get('host')
    user = mysql2_config.get('user')
    password = mysql2_config.get('password')
    database_name = mysql2_config.get('database')
    port = mysql2_config.get('port')
    db = MysqlHelper(host, user, password, database_name,port=port)

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")  # 这个地方可以写域名
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Content-Type' , 'application/json;charset=UTF-8')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')

    ## 查询数据库的任务
    @tornado.gen.coroutine
    def get(self):

        # self.db = MysqlHelper(host, user, password, dbname, port=3313)
        self.db.connect()
        lst = self.db.get_all(sql="select * from task")
        list=[]
        for item in lst:
            dict = {}
            dict['name'] = item[1]
            dict['data'] = item[2]
            dict['startTime'] = item[3]
            dict['endTime'] = item[4]
            dict['algoType'] = item[5]
            dict['algo'] = item[6]
            dict['module'] = item[7]
            dict['status'] = item[8]
            list.append(dict)
        self.write(json.dumps(list,ensure_ascii=False))


    ##创建任务
    @tornado.gen.coroutine
    def post(self):
        jsonbyte = self.request.body
        jsonstr = jsonbyte.decode('utf-8')
        jsonobj = json.loads(jsonstr)
        obj = {}
        obj['name'] = jsonobj.get('name')
        obj['data'] = jsonobj.get('data')
        obj['startTime'] = jsonobj.get('startTime')
        obj['algoType'] = jsonobj.get('algoType')
        obj['algo'] = jsonobj.get('algo')
        obj['module'] = jsonobj.get('module')
        obj['status'] = jsonobj.get('status')
        sql = get_i_sql('task',obj)
        #print(sql)
        n = self.db.insert(sql)
        if(n==1):
            _thread.start_new_thread(self.coreOperation,(obj['algo'],obj['name'],obj['data'],jsonobj.get('beginTime'),jsonobj.get("endTime"),jsonobj.get("module_name")))
        retdata = {}
        retdata['ret'] = True
        retdata['status'] = 0
        self.write(json.dumps(retdata,ensure_ascii=False))

    @tornado.gen.coroutine
    def delete(self,id):
        self.db.connect()
        condition = {}
        condition['id'] = id
        sql = get_d_sql('task')
        self.db.delete(sql)

    # 处理OPTIONS请求
    @tornado.gen.coroutine
    def options(self):
        pass