from datetime import datetime,timedelta
import numpy as np
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
import scipy.sparse as sp
import sys
from tqdm import tqdm
import time
from collections import OrderedDict, Counter
import pymysql
import json

import controller

mod_timepos = {}
mod_timepos['AAA'] = 1
mod_timepos['CMS'] = 2
mod_timepos['CCS'] = 3
mod_timepos['SA'] = 3
mod_timepos['MDU'] = 3
mod_timepos['VOD'] = 3
mod_timepos['PORTAL'] = 2
mod_logpos = {}
mod_logpos['MDU'] = 10
mod_logpos['VOD'] = 10
mod_logpos['CMS'] = 9
mod_logpos['SA'] = 10
mod_logpos['CCS'] = 11
mod_logpos['AAA'] = 8
mod_logpos['PORTAL'] = 9
mod_pattern = {}
mod_pattern['AAA'] = '\[ERROR\] (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) ([^|]+) |'
mod_pattern['CMS'] = '\[ERROR\]\[([^|]+)\] (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) ([^|]+) |'
mod_pattern['CCS'] = 'ERROR (\w+) (\w+) (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) ([^ ]+) - ([^\n]+)'
mod_pattern['SA'] = 'ERROR (\w+) (\w+) (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) - ([^\n]+)'
mod_pattern['MDU'] = 'ERROR (\w+) (\w+) (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) - ([^\n]+)'
mod_pattern['VOD'] = 'ERROR (\w+) (\w+) (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3}) - ([^\n]+)'
mod_pattern['PORTAL'] = '\[ERROR\] \[([^|]+)\] (\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{1,3}) ([^|\n]+) |'
mod_template = {}
# mod_template['AAA'] = {0:'Digest Responses are not matched!',1:'had Failed to Login',2:'Can not find user or frontend',3:'No shared secret!'}
# mod_template['CMS'] = {0:r'##################ERROR DUMP########################',1:r'Request URI:',2:r'Context Path:',3:r'Request URL:',4:r'Query String:',5:'================ HTTP Header ================',6:'accept =',7:'authorization =',8:'username=',9:'content-length =',10:'content-type =',11:'host =',12:'user-agent =',13:'============================================',14:'###  >>> VAP COMMAND = SetView',15:'================ HTTP Parameter ================',16:'displayedCamera =',17:'isDefault =',18:'name =',19:'screenLayout =',20:'###  >>> VAP COMMAND = ReleaseRoute',21:'id =',22:'###  >>> VAP COMMAND = RequestRoute',23:'userNumber =',24:'userIdInCcs =',25:'puId =',26:'cameraId =',27:'requestType =',28:'serviceType =',29:'streamId =',30:'streamType =',31:'mainId =',32:'legId =',33:'Read timed out'}
# mod_template['CCS'] = {0:'InviteComponentInstanceLiveForPu',1:'DoTxResponse decode',2:'SipSCApiComponentNew'}
# mod_template['SA'] = {0:'SIP Message',1:'CDatabaseManager'}
# mod_template['MDU'] = {0:'DoHttpIndication: Can Not Handler'}
# mod_template['VOD'] = {0:'Sip Register error!',1:'TcpRtspTimeSa::Read1'}
# mod_logstruc = {}
# mod_logstruc['MDU'] = {0:'DoHttpIndication: Can Not Handler CmdName = ([\w/_\-\.\:]+)'}
# mod_logstruc['CMS'] = {k:(v+' ([\w/_,\-\.\:]+)') for k,v in mod_template['CMS'].items()}
sys_logpos = {}
sys_logpos['CMS'] = 6
sys_logpos['CCS'] = 6
sys_logpos['PORTAL'] = 6
sys_pattern = {}
sys_pattern['CCS'] = '(\w{3})[ ]{1,2}(\d+) (\d{2}):(\d{2}):(\d{2}) ([^\n]+)'
sys_pattern['CMS'] = '(\w{3})[ ]{1,2}(\d+) (\d{2}):(\d{2}):(\d{2}) ([^\n]+)'
sys_pattern['PORTAL'] = '(\w{3})[ ]{1,2}(\d+) (\d{2}):(\d{2}):(\d{2}) ([^\n]+)'
sys_template = {}
sys_template['CCS'] = {0:'unable to open',1:'Read from socket failed',2:'Authentication failure for root from',3:'User not known to the underlying authentication module for illegal user',4:'trying to get more bytes',5:'buffer_get failed',6:'buffer error',7:'Products Dump',8:'Commandline params',9:"Couldn't resolve host name",10:'Write failed',11:'Argument Dump'}
sys_template['PORTAL'] = {0:'unable to open',1:'Read from socket failed',2:'Authentication failure for root from',3:'User not known to the underlying authentication module for illegal user',4:'trying to get more bytes',5:'buffer_get failed',6:'buffer error',7:'Products Dump',8:'Commandline params',9:"Couldn't resolve host name",10:'Write failed',11:'Argument Dump'}
sys_template['CMS'] = {0:'unable to open',1:'Read from socket failed',2:'Authentication failure for root from',3:'User not known to the underlying authentication module for illegal user',4:'trying to get more bytes',5:'buffer_get failed',6:'buffer error',7:'Products Dump',8:'Commandline params',9:"Couldn't resolve host name"}
sys_timepos = {}
sys_timepos['CMS'] = 1
sys_timepos['CCS'] = 1
sys_timepos['PORTAL'] = 1


# 数据库配置
# host = "localhost"
# root = "root"
# password = "whb6603whb5620"
# database_name = "zhongxin_test"

template_standard = "mod_template"
template_backup = "mod_template_add"

print(os.getcwd())
print('11111')
print(sys.argv[0])
config_json = json.load(open('config.json','r',encoding='utf-8'))
mysql_config = config_json['mysql']

host = mysql_config.get('host',"10.141.208.77")
root = mysql_config.get('user',"root")
password = mysql_config.get('password',"123456")
database_name = mysql_config.get('database',"zhongxin")
port = mysql_config.get('port')

# 判断数据表是否存在
def table_exists(cursor, database, db):

    # 创建表
    sql = "CREATE TABLE IF NOT EXISTS `mod_template` (`eid` int(11) NOT NULL AUTO_INCREMENT, `module_name` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL, `error_code` int(11) DEFAULT NULL, `template` varchar(300) COLLATE utf8mb4_unicode_ci DEFAULT NULL, `regex` varchar(300) COLLATE utf8mb4_unicode_ci DEFAULT NULL, `json_description` varchar(500) COLLATE utf8mb4_unicode_ci DEFAULT NULL, PRIMARY KEY (`eid`), KEY `module_name_index` (`module_name`)) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    try:
        cursor.execute(sql)
        db.commit()
        print("Successfully added table")
    except Exception as e:
        db.rollback()
        print(e)

    return

#读取全部模版到全局变量
def load_template(table_name="mod_template"):
    db = pymysql.connect(host, root, password, database_name,port=port)
    cs = db.cursor()
    cs.execute(f'select module_name, template from {table_name}')
    res = cs.fetchall()
    cs.close()
    db.close()
    global mod_template
    for line in res:
        mdn,tem = line
        mod_template.setdefault(mdn,{})[len(mod_template[mdn])]=tem



# 连接数据库
def database_connection(module_name="CMS", table_name="mod_template", is_train=True, id_need=False, is_used=True):
    # 连接数据库
    db = pymysql.connect(host, root, password, database_name,port=port)
    cs = db.cursor()

    # 判断是否初始化表
    # table_exists(cs, database_name, db)

    # 抽取对应分类模板
    mod_template[module_name] = controller.template_extract(cs, db, module_name=module_name, table_name=table_name, is_train=is_train, id_need=id_need, is_used=is_used)

    db.close()
    return mod_template


# 删除模块
def delete_center(error_code, module_name="CMS", table_name="mod_template", is_used=False):

    db = pymysql.connect(host, root, password, database_name,port=port)
    cs = db.cursor()

    controller.template_delete(cs, db, error_code=error_code, module_name=module_name, table_name=table_name, is_used=is_used)

    db.close()


# 保存模块
def save_center(module_name="CMS", table_name="mod_template", template_name={}, condition=None, is_used=False):

    if is_used:
        is_backup = False
    else:
        is_backup = True

    db = pymysql.connect(host, root, password, database_name,port=port)
    cs = db.cursor()

    controller.template_save(cs, db, module_name=module_name, table_name=table_name, template_name=template_name, is_backup=is_backup, condition=condition)

    db.close()


# 更新模块
def update_center(module_name="CMS", table_name="mod_template", template_name={}, error_code=None, condition=None, is_used=True):

    db = pymysql.connect(host, root, password, database_name,port=port)
    cs = db.cursor()

    controller.template_update(cs, db, module_name=module_name, table_name=table_name, template_name=template_name, error_code=error_code, condition=condition, is_used=is_used)

    db.close()

###mod_template写入数据库
def write_module_template(db):
    table_name = 'mod_template'
    cur = db.cursor()
    query = f'insert into {table_name}()'


###系统日志格式匹配
def match_system_log(log, system_name='CMS'):
    pat = sys_pattern[system_name.upper()]
    return re.match(pat, log)


# logstruc获取
def logstruct(module_name="CMS"):
    mod_logstruc = {}
    mod_template = database_connection(module_name=module_name, table_name="mod_template", is_train=False)

    mod_logstruc['MDU'] = {0:'DoHttpIndication: Can Not Handler CmdName = ([\w/_\-\.\:]+)'}
    if module_name == 'CMS':
        mod_logstruc['CMS'] = {k:(v[1]+' ([\w/_,\-\.\:]+)') for k,v in enumerate(mod_template['CMS'].items())}

    return mod_logstruc[module_name]


###模块日志中参数计数
def para_count(logs, module_name):
    template = database_connection(module_name=module_name.upper(), table_name="mod_template", is_train=False)
    logstruc = logstruct(module_name.upper())
    cnt = {}
    for log in tqdm(logs):
        for k, v in logstruc.items():
            res = re.match(v, log)
            if (not res is None) and (res.group(1) != ''):
                para = res.group(1)
                cnt.setdefault(k, {}).setdefault(para, 0)
                cnt[k][para] += 1
                break
    return cnt


###系统日志格式匹配
def match_log(log, module_name='aaa'):
    pat = mod_pattern[module_name.upper()]
    return re.match(pat, log)


###高频词计算
def counter(logs, min_df=5):
    cnt = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=min_df)
    res = cnt.fit_transform(logs)
    return cnt, res


###原生Counter词频统计
def buildin_counter(logs, min_df=5):
    word_cnt = Counter([])
    for log in logs:
        tmp = Counter(re.split('[ ,|;]', log))
        word_cnt += tmp
    return {k: v for k, v in word_cnt.items() if v >= min_df}


###系统日志从文本变0,1,2,...的数字序列化
def label_sys_logs(logs, system_name='cms'):
    template = {}
    template = sys_template[system_name.upper()]

    labels = []
    for log in logs:
        label = len(template)
        for tem in template.items():
            if log.find(tem[1]) >= 0:
                label = tem[0]
                break
        labels.append(label)
    return labels


###模块日志从文本变0,1,2,...的数字序列化
def label_logs(logs, module_name='aaa'):
    template = {}
    template = mod_template[module_name.upper()]

    labels = []
    for log in logs:
        label = len(template)
        for tem in template.items():
            if log.find(tem[1]) >= 0:
                label = tem[0]
                break
        labels.append(label)
    return labels


###日志聚类
def cluster_log(logs, thres=1.0, method='kmeans', args={'n_cluster': 4}, n_clusters=20):
    if method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', distance_threshold=thres)
    else:
        model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=100, tol=0.01, verbose=1, n_init=3)
    if sp.issparse(logs):
        logs = logs.todense()
    label = model.fit(logs)
    return model, label


###日志聚类
def label_logs_hb(logs, module_name='AAA', template=None):

    db = pymysql.connect(host, root, password, database_name,port=port)
    cs = db.cursor()

    if template is None:
        template = {}
    labels = []
    extra_labels = set()
    tem_idx = 0
    for log in logs:
        label = len(template)
        for tem in template.items():
            if log.find(tem[1]) >= 0:
                label = tem[0]
                break
        labels.append(label)
        if label == len(template):
            extra_labels.add(log)
            tem_idx += 1

###########################2
    if len(extra_labels) == 0:
        return labels, {}

    # 对unk分类向量化
    max_features = 20000
    if tem_idx < 40:
        n_clusters = tem_idx
    else:
        n_clusters = 40

    vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, encoding='latin-1')
    labels_list_origin = list(extra_labels)


    # 修剪数组内字符串长度
    labels_list = [item[0:80] for item in labels_list_origin]

    X = vectorizer.fit_transform((d for d in labels_list))

    # 对unk分类进行聚类
    model, results = cluster_log(X, n_clusters=n_clusters, method="kmeans")

    # 打印导出每个聚类簇
    res2series = pd.Series(results.labels_)
    label_max = np.max(results.labels_)
    label_idx = 1
    extra_list = []

    # print("新提取到的日志数据类型：")
    for i in range(label_max + 1):
        res = res2series[res2series.values == i]
        if len(res.index) > 0:
            extra_list.append(labels_list_origin[res.index[0]])

            # print("类别为" + str(label_idx) + "的数据:")
            # label_idx += 1
            # list_idx = 0
            # for j in res.index:
            #     if list_idx > 10:
            #         break
            #     print(labels_list_origin[j])
            #     list_idx += 1

    extra_list = {idx: items for idx, items in enumerate(extra_list)}

    # 存入抽取新分类模型
    controller.template_save(cs, db, module_name=module_name, table_name="mod_template", template_name=extra_list, is_backup=True)

    db.close()

    return labels, extra_list


###系统日志模版计数
def template_sys_cnt(logs, system_name='cms'):
    template = {}
    template = sys_template[system_name.upper()]

    cnt = {k: 0 for k in template.values()}
    for log in logs:
        label = len(template)
        for tem in template.items():
            if log.find(tem[1]) >= 0:
                cnt[tem[1]] += 1
                break
    return cnt


###模块日志模版计数
def template_cnt(logs, module_name='aaa', template=None):
    if template is None:
        template = mod_template[module_name.upper()]

    cnt = {k: 0 for k in template.values()}
    for log in logs:
        label = len(template)
        for tem in template.items():
            if log.find(tem[1]) >= 0:
                cnt[tem[1]] += 1
                break
    return cnt


###中间数据写出
def write_data(data, labels, file_name):
    with open(file_name, 'w') as f:
        for i, d in enumerate(tqdm(data)):
            f.write(str(labels[i]) + '\t' + d + '\n')


###系统日志时间解析
def parse_sys_data_time(data, system_name='cms', week_day=False):
    times = []
    pattern = sys_pattern[system_name.upper()]
    weekdays = []

    for d in data:
        res = re.match(pattern, d)
        start = res.span(sys_timepos[system_name.upper()])[0]
        t = d[start:start + 15]
        t = "2019 " + t
        # timeArray = time.strptime(t, "%b %d %H:%M:%S")
        timeArray = time.strptime(t, '%Y %b %d %H:%M:%S')
        ts = int(time.mktime(timeArray))

        weekdays.append(datetime.fromtimestamp(ts).weekday())
        times.append(ts)
    if week_day:
        return times, weekdays
    return times


###模块日志时间解析
def parse_data_time(data, module_name='aaa', week_day=False):
    times = []
    pattern = mod_pattern[module_name.upper()]
    weekdays = []

    for d in data:
        res = re.match(pattern, d)
        start = res.span(mod_timepos[module_name.upper()])[0]
        t = d[start:start + 19]
        timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
        ts = int(time.mktime(timeArray))
        weekdays.append(datetime.fromtimestamp(ts).weekday())
        times.append(ts)
    if week_day:
        return times, weekdays
    return times


###系统日志时间统计
def data_time_sys_counter(data, system_name='cms'):
    times = []
    pattern = sys_pattern[system_name.upper()]
    cnt = OrderedDict()

    for d in data:
        res = re.match(pattern, d)
        start = res.span(sys_timepos[system_name.upper()])[0]
        t = d[start:start + 6]
        cnt[t] = cnt.setdefault(t, 0) + 1
    return cnt


###模块日志时间统计
def data_time_counter(data, module_name='aaa'):
    times = []
    pattern = mod_pattern[module_name.upper()]
    cnt = OrderedDict()

    for d in data:
        res = re.match(pattern, d)
        start = res.span(mod_timepos[module_name.upper()])[0]
        t = d[start:start + 10]
        cnt[t] = cnt.setdefault(t, 0) + 1
    return cnt


###系统日志，往异常0,1,...序列插入非异常，插入后，原异常序列数值加1，0代表非异常
def labels_sys_sequence(labels, timestamps):
    if labels == []:
        return []
    m = max(timestamps)
    n = min(timestamps)

    new_l = []
    new_l.append(labels[0] + 1)
    for i in range(1, len(timestamps)):
        td = int(timestamps[i] - timestamps[i - 1] - 1)
        while td < -1:
            td %= 31536000
        new_l += [0] * max(td, 0)
        new_l.append(labels[i] + 1)
    return new_l


###模块日志，往异常0,1,...序列插入非异常，插入后，原异常序列数值加1，0代表非异常
def labels_sequence(labels, timestamps):
    if labels == []:
        return []
    new_l = []
    new_l.append(labels[0] + 1)
    for i in range(1, len(timestamps)):
        new_l += [0] * max(int(timestamps[i] - timestamps[i - 1] - 1), 0)
        new_l.append(labels[i] + 1)
    return new_l


###从时间戳计算时间差
def timediff(timestamps):
    if timestamps == []:
        return []
    dts = [0]
    for i in range(1, len(timestamps)):
        dts.append(timestamps[i] - timestamps[i - 1])
    return dts


###负采样非异常
def neg_sample(timestamps, week_day=False):
    if timestamps == []:
        return []
    neg = []
    weekdays = []
    for i in range(1, len(timestamps)):
        d = timestamps[i] - timestamps[i - 1]
        if d <= 0:
            continue
        delt = np.random.randint(0, d)
        neg.append(delt)
        wd = datetime.fromtimestamp(timestamps[i] + delt).weekday()
        # ng_wd = wd
        # while ng_wd == wd:
        # ng_wd = np.random.randint(0,6)
        ng_wd = np.random.randint(0, 6)
        weekdays.append(ng_wd)
    if week_day:
        return neg, weekdays
    return neg


###0,1,2,...的异常码转为模版，注意如果异常码从1开始，0是非异常的话，输入的label要减1
def label2error(label, module_name='MDU', template={}):
    # template = mod_template[module_name.upper()]
    # temp = database_connection(module_name=module_name.upper(), is_train=False)
    # template = temp[module_name]
    if label in template:
        return template[label]
    else:
        return "<unknown> error"

load_template()

def timestamp2iso(timestamp, format='%Y-%m-%dT%H:%M:%S.%fZ'):
    """
    时间戳转换到ISO8601标准时间(支持微秒级输出 YYYY-MM-DD HH:MM:SS.mmmmmm)
    :param timestamp:时间戳，支持 秒，毫秒，微秒级别
    :param format:输出的时间格式  默认 iso=%Y-%m-%dT%H:%M:%S.%fZ；其中%f表示微秒6位长度

    此函数特殊处理，毫秒/微秒部分 让其支持该部分的字符格式输出
    :return:
    """
    # 显示成东8区时间
    timestamp = timestamp+8*3600*1000
    format = format.replace('%f','{-FF-}')#订单处理微秒数据 %f
    length = min(16, len(str(timestamp)))#最多去到微秒级

    #获取毫秒/微秒 数据
    sec = '0'
    if length != 10:#非秒级
        sec = str(timestamp)[:16][-(length - 10):]#最长截取16位长度 再取最后毫秒/微秒数据
    sec = '{:0<6}'.format(sec)#长度位6，靠左剩下的用0补齐
    timestamp = float(str(timestamp)[:10])#转换为秒级时间戳
    return datetime.utcfromtimestamp(timestamp).strftime(format).replace('{-FF-}',sec)

def getEveryDay(begin_date,end_date):
    # 前闭后闭
    date_list = []
    begin_date = datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y.%m.%d")
        date_list.append(date_str)
        begin_date += timedelta(days=1)
    return date_list

if __name__ == '__main__':
    #mod_template = database_connection()
    sa_sample = 'ERROR root 0x7f74e1292700 2020-03-12 12:30:51,121 - SIP Message [MSG_GET_VCU_RECORD_INFO_RSP] ResponseError: no PAGE_INFO' 
    #aaa_sample = '[ERROR] 2020-03-06 04:22:24,282  | com.zxelec.viss.aaa.radius.handler.CcsAuthHandler.handle(407)'
    # res = re.match(sa_pattern,sa_sample)
    # print(res.group())
    # print(res.group(11))
    pass

