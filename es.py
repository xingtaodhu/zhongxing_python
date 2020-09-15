import time
from datetime import datetime
import os
from elasticsearch import Elasticsearch
import csv
import  datetime
from utils import timestamp2iso,getEveryDay
import json

#上线需要修改
config_json = json.load(open('config.json','r',encoding='utf-8'))
es_config = config_json['es']
host = es_config.get('host')
port = es_config.get('port')
es = Elasticsearch(hosts=[{"host":host,"port":port}])

def is_indexExits(index):
    return es.indices.exists(index=index)

def query(index,startTime,endTime,module_name="aaa"):
    module_name = module_name.lower()
    query_json = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": startTime,
                                    "lte": endTime
                                }
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {
                                        "prefix": {
                                            "message.keyword": {
                                                "value": "ERROR"
                                            }
                                        }
                                    },
                                    {
                                        "prefix": {
                                            "message.keyword": {
                                                "value": "[ERROR"
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "term":{
                                "app_id":module_name
                            }
                        }
                    ]
                }
            },
            "_source": [
                "@timestamp",
                "message",
                "app_id"
            ],
            "sort": [
                {
                    "@timestamp": {
                        "order": "asc"
                    }
                }
            ]
        }
    res = es.search(index = index, scroll = '5m',size = 1000,body = query_json)
    print("Got %d Hits:" % res['hits']['total']['value'])
    results = res['hits']['hits']  # es查询出的结果第一页
    total = res['hits']['total']['value'] # es查询出的结果总量
    scroll_id = res['_scroll_id']  # 游标用于输出es查询出的所有结果

    for i in range(0, int(total / 100) + 1):
        # scroll参数必须指定否则会报错
        query_scroll = es.scroll(scroll_id=scroll_id, scroll='5m')['hits']['hits']
        results += query_scroll
    cur_time=datetime.datetime.now()
    cur_time=cur_time.strftime('%Y-%m-%d %H:%M:%S')

    path = '/Users/xingtao/upload'
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    f=[]
    with open(path+'/ES'+cur_time+'.csv', 'w', newline='', encoding='utf-8') as flow:
        csv_writer = csv.writer(flow)
        for res in results:
            f.append(res['_source']['message'])
            csv_writer.writerow([res['_source']['message']])
    return f




if __name__ == '__main__':
    # # dateArray = datetime.datetime.utcfromtimestamp(1595821055)
    # time_local = time.localtime(1595821055)
    # startTime = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    # print(getEveryDay(startTime.split(' ')[0], '2020-09-19'))
    print(is_indexExits('vkit123-2020.07.08'))
    # print(startTime)
    # print(timestamp2iso(1595821055088))
    #print(datetime.datetime.now().replace(microsecond=0).isoformat())
    #query('vkit123-2020.07.08','2020-07-08T08:04:00.103Z','2020-07-08T19:16:00.103Z')