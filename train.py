# -*- coding: utf-8 -*-


from concurrent.futures import ThreadPoolExecutor

from tornado.concurrent import run_on_executor
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import json
import traceback
from svm import svm
from lgbm import lightgbm
from taskHandler import TaskHandler
from predictHandler import PredictHandler


def add(a, b):
    c = int(a) + int(b)
    return str(c)


class MainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(32)

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")  # 这个地方可以写域名
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    @tornado.gen.coroutine
    def get(self):
        '''get接口'''
        htmlStr = '''
                    <!DOCTYPE HTML><html>
                    <meta charset="utf-8">
                    <head><title>Get page</title></head>
                    <body>
                    <form		action="/train"	method="post" >
                    <input type="hidden"      name ="a"   value="2"  />
                    <input type="hidden"      name ="b"  value="3"   />
                    训练文件路径:<input type = "text" name="file" value="./train1.csv" /><br>
                    模型:<select name='module'>

                             <option value='svm'>SVM</option>

                            <option value='lightgbm'>LightGBM</option>

                    </select><br>

                    <input type="submit"	value="开始训练"	/>
                    </form></body> </html>
                '''
        self.write(htmlStr)

    @tornado.gen.coroutine
    def post(self):
        '''post接口， 获取参数'''
        # a = self.get_argument("a", None)
        # b = self.get_argument("b", None)
        module = self.get_argument("module",None)
        yield self.coreOperation(module)

    @run_on_executor
    def coreOperation(self,module):
        '''主函数'''
        try:
            if module == 'svm':
                precision,recall = svm()
                result = precision+','+recall
            if module == 'lightgbm':
                precision,recall = lightgbm()

                result = str(precision)+','+str(recall)

            # if a != '' and b != '':
            #     result = add(a, b)  # 可调用其他接口
            #     if result:
            #         result = json.dumps({'code': 200, 'result': result, })
            #     else:
            #         result = json.dumps({'code': 210, 'result': 'no result', })

            else:
                result = json.dumps({'code': 211, 'result': 'wrong parameter', })
            self.write(result)
        except Exception:
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            result = json.dumps({'code': 503, 'result':'error'})
            self.write(result)


if __name__ == "__main__":
    # dataset = np.loadtxt("train1.csv", delimiter=',', encoding='utf-8')
    # # df = pd.DataFrame(dataset)
    # # df[df.columns[-1]] = df[df.columns[-1]].shift(-1)
    # # dataset = df.values
    #
    # label = dataset[:, -1]
    # print(label.shape)
    # feature = dataset[:, :-1]
    # print(label.shape)
    #
    # x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=0)
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r'/train', MainHandler),(r'/task',TaskHandler),(r'/predict',PredictHandler)], autoreload=False, debug=False)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8832)
    tornado.ioloop.IOLoop.instance().start()
