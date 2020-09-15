import copy

import pickle as pkl
import time
import json
import numpy as np
import utils
import utils2
from tqdm import tqdm,trange
from sklearn import metrics
import xgboost as xgb
import sys
from datetime import datetime

data_path = {}
data_path['VOD'] = '/home/zhangtianqi/project/zhongxin/data3/infocollection/modulelog/SA、VOD、MDU_43.27.135/130/vod_log.log'
data_path['MDU'] = '/home/zhangtianqi/project/zhongxin/data4/logs/模块日志/MDU/43.27.134.130/log.log'
data_path['CMS'] = '/home/zhangtianqi/project/zhongxin/data4.5/cms.log'
data_path['CCS'] = '/home/zhangtianqi/project/zhongxin/data5/ccs/运行日志/log.log'
data_path['PORTAL'] = '/home/zhangtianqi/project/zhongxin/data5/portal/运行日志/portal.log'
data_path['SA'] = '/home/zhangtianqi/project/zhongxin/data5/集群1运行/log/sa/sa_130/log.log'
data_path['AAA'] = '/home/zhangtianqi/project/zhongxin/data5/cms/aaa运行日志/viss-aaa.log'

# 模块的日志分片数
data_slice = {}
data_slice['VOD'] = 1
data_slice['MDU'] = 1
data_slice['CMS'] = 3
data_slice['CCS'] = 10
data_slice['PORTAL'] = 100
data_slice['SA'] = 14
data_slice['AAA'] = 24

def load_data_from_file(module_name,file_name):
    logs = []
    data = []
    module_name = module_name.upper()
    module_log_pos = utils.mod_logpos[module_name.upper()]

    # for i in range(data_slice[module_name.upper()]-1, -1, -1):
    #     file_name = data_path[module_name.upper()]
    #     if i > 0:
    # file_name += '.' + str(i)
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            res = utils.match_log(line, module_name)
            if (not (res is None)) and (not (res.group() == '')):
                data.append(res.group())
                logs.append(res.group(module_log_pos))
    return data, logs


def test_raw_xgb(logs_raw,module_name,time_interval=300,is_bin=False):
    '''
    params:
        module_name: string 模块名比如aaa,大小写都可以
        is_train: bool 是否训练
        time_interval: int 预测间隔
        is_bin: bool 忽略此参数保持False 二分类True 多分类False
    out:
        res: dict，测试集指标 例如{"accuracy":ac,"recall":rc,"confusion_matrix":cf.tolist()}
    '''
    ###load request
    #data_raw = request.data
    #data_raw = json.loads(data_raw)
    #logs_data = data_raw['logs']
    #module_name = data_raw['module_name'].upper()
    #time_interval = data_raw['interval'] if 'interval' in data_raw else 300
    #is_train = data_raw['is_train'] if 'is_train' in data_raw else False
    #is_bin = data_raw.get('is_bin')
    #is_bin = False if is_bin is None else True
    module_name = module_name.upper()

    logs = []
    data = []
    res = {}
    module_name = module_name.upper()
    module_log_pos = utils.mod_logpos[module_name.upper()]

    for line in tqdm(logs_raw):
        res = utils.match_log(line, module_name)
        if (not (res is None)) and (not (res.group() == '')):
            data.append(res.group())
            logs.append(res.group(module_log_pos))
    ###parse logs
    # data, logs = load_data_from_file(module_name,file_name)
    num_labels = len(utils.mod_template[module_name.upper()]) + 2
    labels = utils.label_logs(logs,module_name)
#   utils.write_data(data,labels,f'data/{module_name.lower()}.csv')
    timestamps,weekdays = utils.parse_data_time(data,module_name,True)
    timediffs = utils.timediff(timestamps)
    neg_diffs,neg_week = utils.neg_sample(timestamps,True)
    neg_label = [0]*int(len(neg_diffs))
    
    weekdays.extend(neg_week)
    x = timediffs
    x.extend(neg_diffs)
    x = [[x[i],weekdays[i]] for i in range(len(x))]
    y = [x+1 for x in labels]
    y.extend(neg_label)
    if is_bin:
        num_labels = 2
        y = [0 if iii==0 else 1 for iii in y]
    xy = np.hstack([np.array(x),np.array(y)[:,np.newaxis]])
    np.random.shuffle(xy)
    x = xy[:,0:2]
    y = xy[:,2][:,np.newaxis]
    #labels_seq = utils.labels_sequence(labels,timestamps)
    
    train_idx_end = int(len(x)*0.5)
    x_train = x[:train_idx_end]
    y_train = y[:train_idx_end]
    x_test = x[train_idx_end:]
    y_test = y[train_idx_end:]
    print('标签分布',np.bincount(y_train.flatten()))
    #x_train = torch.tensor(x_train).float()
    #x_test = torch.tensor(x_test).float()
    #y_train = torch.tensor(y_train).float()
    #y_test = torch.tensor(y_test).float()
    #feats_dim = len(x[0])
    lr = 1
    data_train = xgb.DMatrix(x_train,label=y_train)
    data_test = xgb.DMatrix(x_test)
    params = {'num_class':num_labels,'max_depth': 20,'min_child_weight':4, 'eta': lr, 'objective': 'multi:softprob','verbosity':3}
    print('training...',file=sys.stderr)
    bst = xgb.train(params, data_train, 100)
    pkl.dump(bst,open(f'xgb.{module_name}.{time_interval}.bat','wb'))
    print('testing...',file=sys.stderr)
    p_list = bst.predict(data_test)
    y_list = np.argmax(p_list,axis=1)
    ac = metrics.accuracy_score(y_test,y_list)
    rc = metrics.recall_score(y_test,y_list,average='macro')
    cf = metrics.confusion_matrix(y_test,y_list) 
    print(cf)
    print(f'accuracy: {ac}, recall: {rc}')

    res = {"accuracy":ac,"recall":rc,"confusion_matrix":cf.tolist()}

    return json.dumps(res)


def predict_raw_xgb(logs_raw, start_time, end_time, module_name, is_train=False, time_interval=300, is_bin=False):
    logs = []
    data = []
    res = {}
    module_name = module_name.upper()
    module_log_pos = utils.mod_logpos[module_name.upper()]

    for line in tqdm(logs_raw):
        res = utils.match_log(line, module_name)
        if (not (res is None)) and (not (res.group() == '')):
            data.append(res.group())
            logs.append(res.group(module_log_pos))

    module_name = module_name.upper()
    num_labels = len(utils.mod_template[module_name.upper()]) + 2
    labels = utils.label_logs(logs, module_name)
    #   utils.write_data(data,labels,f'data/{module_name.lower()}.csv')
    timestamps, weekdays = utils.parse_data_time(data, module_name, True)
    start_timestamp = None
    end_timestamp = None
    if not start_time is None:
        start_time_array = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        start_timestamp = int(time.mktime(start_time_array))


    if not end_time is None:
        end_time_array = time.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        end_timestamp = int(time.mktime(end_time_array))
    cur = end_timestamp + 1
    try:
        delta_p = max(cur - 1 - timestamps[-1], 0)
    except:
        delta_p = 0
    x_stamp = list(range(1, 1 + time_interval))
    x_stamp = [xxx + delta_p for xxx in x_stamp]
    x_weekday = [datetime.now().weekday()] * len(x_stamp)

    x_test = np.hstack([np.array(x_stamp)[:, np.newaxis], np.array(x_weekday)[:, np.newaxis]])
    # labels_seq = utils.labels_sequence(labels,timestamps)
    bst = pkl.load(open(f'xgb.{module_name}.{time_interval}.bat', 'rb'))

    lr = 1
    ans = []
    pre_e = -1
    y_list = []
    for i in range(len(x_test)):
        if pre_e >= 0:
            data_test = xgb.DMatrix(np.array([[x_test[i][0] - x_test[pre_e][0], x_test[i][1]]]))
        else:
            data_test = xgb.DMatrix(np.array([x_test[i]]))
        p_list = bst.predict(data_test)
        y_p = np.argmax(p_list, axis=1).flatten()[0]
        if y_p > 0:
            pre_e = i
        y_list.append(y_p)
    ye = [[idx, ec] for idx, ec in enumerate(y_list) if ec > 0]
    origin = utils2.database_connection(module_name=module_name, table_name="mod_template", is_train='predict', is_used=True)

    origin_list = {}
    origin_list[module_name] = {idx: row["template"] for idx, row in enumerate(origin[module_name])}
    error_template = {}
    error_template[module_name] = {idx: row["template"] for idx, row in enumerate(origin[module_name])}
    errors = []
    for idx, ec in ye:
        tts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cur + idx))
        error = utils2.label2error(ec - 1, module_name, error_template[module_name])
        errors.append([tts, int (ec - 1), error])
    res = {}
    res['errors'] = errors
    # print(res)
    return json.dumps(res)


if __name__ == '__main__':
    test_raw_xgb("./43.27.134.36_viss-aaa.log","aaa")