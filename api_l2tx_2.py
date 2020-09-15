# coding=utf-8

import copy

import re
import time
import sys
import os
from urllib import parse
import json
import numpy as np
import utils2
from tqdm import tqdm,trange
import model
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics


#config
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
hidden_dim = 16
num = {}
mdls = {}
hiddens = {}
lr = 0.001
ce = nn.CrossEntropyLoss()
optims = {}


#######################################
def get_model(module_name='MDU', is_train=True):
    # 两种情况：1.如果之前没有模型 2.想要训练  满足一个就需要重新搭建模型
    if module_name not in mdls or is_train:
        if is_train:
            state = "train"
        else:
            state = "predict"
        temp = utils2.database_connection(module_name=module_name.upper(), is_train=state, is_used=True)
        num_labels = len(temp[module_name]) + 1
        # num_labels = len(utils2.mod_template[module_name.upper()])
        md = model.latent_interval_cross(hidden_dim=hidden_dim, num_labels=num_labels).to(device)

        hid = torch.zeros(1, hidden_dim)
        # 不需要重新训练的时候就可以使用之前的参数
        if not is_train:
            # os.path.isfile('./chk/{module_name.lower()}_tr.chk')
            h__s = torch.load(f'./chk/{module_name.lower()}_tr.chk', map_location='cpu')
            md.load_state_dict(h__s)
            # hid = h__s['hidden']

        opt = optim.Adam(md.parameters(), lr=lr)
        mdls[module_name.upper()] = md
        hiddens[module_name.upper()] = hid
        optims[module_name.upper()] = opt
        num[module_name.upper()] = num_labels
    else:
        md = mdls[module_name.upper()]
        hid = hiddens[module_name.upper()]
        opt = optims[module_name.upper()]
        # num_labels = num[module_name.upper()]

    return md, hid, opt


# def cnt_modu():
#     try:
#         res = {}
#         bd = request.data
#         bd = json.loads(bd)
#         rlogs = bd['logs']
#         for mdname in utils2.mod_template.keys():
#             if not mdname in rlogs:
#                 continue
#             data, logs = utils2.raw2dl(rlogs[mdname],mdname)
#             cnt = utils2.template_cnt_wi(logs,mdname)
#             res[mdname] = cnt
#         return json.dumps(res)
#     except Exception as e:
#         print(e,file=sys.stderr)
#         return json.dumps({})

def test_raw(logs_data,start_time,end_time,module_name,is_train,time_interval=300):
    res = {}
    if not os.path.exists('chk'):
        os.mkdir('chk')
    try:
        ###load request
#        data_raw = request.data
#        data_raw = json.loads(data_raw)
#        logs_data = data_raw['logs'] if 'logs_data' in data_raw else []
#        start_time = data_raw['start_time'] if 'start_time' in data_raw else None
#        end_time = data_raw['end_time'] if 'end_time' in data_raw else None
#        module_name = data_raw['module_name'].upper()
#        time_interval = data_raw['interval'] if 'interval' in data_raw else 300
        module_name = module_name.upper()
        # # 外部给定地址输入
        # data_path = data_raw['data_path'] if 'data_path' in data_raw else 'empty'
        # data_slice = data_raw['data_slice'] if 'data_slice' in data_raw else 0

        ###parse logs
        data = []
        logs = []
        module_log_pos = utils2.mod_logpos[module_name.upper()]


##############
        # is_train = data_raw['is_train'] if 'is_train' in data_raw else true
#        is_train = data_raw['is_train'] if 'is_train' in data_raw else False

        if is_train:
            state = "train"
        else:
            state = "predict"

        origin = utils2.database_connection(module_name=module_name, table_name="mod_template", is_train=state, is_used=True)

        origin_list = {}
        origin_list[module_name] = {idx: row["template"] for idx, row in enumerate(origin[module_name])}

        ###parse time
        start_timestamp = None
        end_timestamp = None
        if not start_time is None:
            start_time_array = time.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            start_timestamp = int(time.mktime(start_time_array))
        if not end_time is None:
            end_time_array = time.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            end_timestamp = int(time.mktime(end_time_array))
        
        # if data_path != 'empty':
        #     for i in range(data_slice - 1, -1, -1):
        #         file_name = data_path
        #         if i > 0:
        #             file_name += '.' + str(i)
        #         with open(file_name, 'r', encoding='utf-8') as f:
        #             for line in tqdm(f):
        #                 res = utils2.match_log(line, module_name)
        #                 if (not (res is None)) and (not (res.group() == '')):
        #                     data.append(res.group())
        #                     logs.append(res.group(module_log_pos))
        #
        # else:
        #     for line in tqdm(logs_data):
        #         res_l = utils2.match_log(line,module_name)
        #         if (not (res_l is None)) and (not (res_l.group() == '')):
        #             data.append(res_l.group())
        #             logs.append(res_l.group(module_log_pos))

        for line in tqdm(logs_data):
            res_l = utils2.match_log(line,module_name)
            if (not (res_l is None)) and (not (res_l.group() == '')):
                data.append(res_l.group())
                logs.append(res_l.group(module_log_pos))

        # 获得模型，分为是否训练两种情况
        md, hid, opt = get_model(module_name, is_train)

###################2
        labels, extra_list = utils2.label_logs_hb(logs, module_name, origin_list[module_name])
        timestamps = utils2.parse_data_time(data, module_name=module_name)

        if (not start_timestamp is None) and (len(timestamps) == 0 or (start_timestamp < timestamps[0])):
            labels = [-1] + labels
            timestamps = [start_timestamp] + timestamps
        if (not end_timestamp is None) and (end_timestamp != timestamps[-1]):
            labels.append(-1)
            timestamps.append(end_timestamp)
        labels_seq = utils2.labels_sequence(labels, timestamps)
        # print(len(labels_seq))
             
        ###train for redundent logs
        if is_train and len(labels_seq) > time_interval:  
            x_test = torch.tensor(labels_seq[:len(labels_seq)-time_interval]).long().to(device)
            y_test = torch.tensor(labels_seq[time_interval:]).long().to(device)
            with trange(x_test.size()[0]) as bar2:
                for i in bar2:
                    out, hid = md(x_test[[i]],hid.detach())
                    loss = ce(out,y_test[[i]])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            torch.save(md.state_dict(), f'./chk/{module_name.lower()}_tr.chk')

        ###predict
        if not is_train:
            x_test2 = torch.tensor(labels_seq[max(0, len(labels_seq)-time_interval):]).long().to(device)
        else:
            x_test2 = torch.tensor(labels_seq[:len(labels_seq) - time_interval]).long().to(device)
        y_pred = []
        errors = []
        # error_template_origin = utils2.database_connection(module_name=module_name, table_name="mod_template", is_train=state)

        error_template = {}
        error_template[module_name] = {idx: row["template"] for idx, row in enumerate(origin[module_name])}

        with trange(x_test2.size()[0]) as bar2:
            for i in bar2:
                out, hid = md(x_test2[[i]], hid.detach())
                y_pred.append(torch.argmax(out).to(torch.device('cpu')).item())
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                continue
            else:
                tts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_timestamp+i+1))
                error = utils2.label2error(y_pred[i]-1, module_name, error_template[module_name])
                errors.append([tts, y_pred[i]-1, error])
        
        if is_train:
            print(len(y_test),len(y_pred))
            ac = metrics.accuracy_score(y_test,y_pred)
            rc = metrics.recall_score(y_test,y_pred,average='macro')
            cf = metrics.confusion_matrix(y_test,y_pred)
            print(cf)
            print(f'accuracy: {ac}, recall: {rc}')

            res = {"accuracy":ac,"recall":rc,"confusion_matrix":cf.tolist()}   
        else:
            res['errors'] = errors

        # 新抽取的模板分类后放入new中
        #res['new'] = extra_list  # {0:xxxxxx, 1:xxxxxxxx, 2:xxxxxxx}

    except Exception as e:
        print(e, file=sys.stderr)

    return json.dumps(res)


############################
# 展示模板的相似度结果
#@app.route('/test_similarity', methods=['POST', 'GET'])
#def test_similarity():
#    ###load request
#    data_raw = request.data
#    data_raw = json.loads(data_raw)
#    module_name = data_raw['module_name'].upper()
#    # 存储匹配信息
#    test = Similarity()
#    sim_set = {}
#
#    try:
#        # 从数据库取出该类型全部模板
#        origin_list = copy.deepcopy(
#            utils2.database_connection(module_name=module_name, table_name="mod_template", is_train="get", is_used=True))
#        extra_list = copy.deepcopy(
#            utils2.database_connection(module_name=module_name, table_name="mod_template", is_train="get", is_used=False))
#
#        for it in extra_list[module_name]:
#            # 分类模板进行剪切，以免太长
#            _it = it['template'][:100]
#            sim_set[_it] = {}
#            for item in origin_list[module_name]:
#                _item = item['template'][:100]
#                sim = test.levenshtein(_item, _it)
#                if sim >= 0.5:
#                    sim_set[_it][_item] = sim
#
#    except Exception as e:
#        print(e, file=sys.stderr)
#
#    # 返回的sim_set中是全部匹配的相关性系数
#    return json.dumps(sim_set)
#
#
## 删除模板
##############################2
#@app.route('/delete/<table_name>', methods=['POST', 'GET'])
#def delete_template(table_name):
#
#    data_raw = request.data
#    data_raw = json.loads(data_raw)
#    error_code = data_raw['error_code']
#    module_name = data_raw['module_name'].upper()
#    is_used = data_raw['is_used']
#
#    try:
#        if error_code.upper() == "ALL":
#            utils2.delete_center(error_code=None, module_name=module_name, table_name=table_name, is_used=is_used)
#
#        else:
#            utils2.delete_center(error_code=error_code, module_name=module_name, table_name=table_name, is_used=is_used)
#
#    except Exception as e:
#        print(e)
#
#    return "success"
#
#
## 查询模板
#@app.route('/get/<table_name>', methods=['POST', 'GET'])
#def get_template(table_name):
#
#    data_raw = request.data
#    data_raw = json.loads(data_raw)
#    module_name = data_raw['module_name'].upper() if 'module_name' in data_raw else "ALL"
#    is_used = data_raw['is_used'] if 'is_used' in data_raw else None
#
#    res = {}
#
#    try:
#        if module_name == "ALL":
#            res = utils2.database_connection(module_name="ALL", table_name=table_name, is_train="get", is_used=is_used)
#        else:
#            res = utils2.database_connection(module_name=module_name, table_name=table_name, is_train="get", is_used=is_used)
#    except Exception as e:
#        print(e)
#
#    res_list = []
#    for item in res.items():
#        for it in item[1]:
#            temp = it['date']
#            if temp != None:
#                it['date'] = temp.strftime("%Y-%m-%d %H:%M:%S")
#            res_list.append(it)
#
#    # 返回结果
#    return json.dumps(res_list)
#
#
## 增加模板
#@app.route('/save/<table_name>', methods=['POST', 'GET'])
#def save_template(table_name):
#
#    data_raw = request.data
#    data_raw = json.loads(data_raw)
#    module_name = data_raw['module_name'].upper()
#    # table_name = data_raw['table_name']
#    template_name = data_raw['template_name']
#    regex = data_raw['regex'] if 'regex' in data_raw else None
#    json_description = data_raw['json_description'] if 'json_description' in data_raw else None
#    is_used = data_raw['is_used']
#
#    condition = {}
#    condition['regex'] = regex
#    condition['json_description'] = json_description
#
#    try:
#        utils2.save_center(module_name=module_name, table_name=table_name, template_name={0: template_name}, condition=condition, is_used=is_used)
#    except Exception as e:
#        print(e)
#
#    return "success"
#
#
## 更新模板
#@app.route('/update/<table_name>', methods=['POST', 'GET'])
#def update_template(table_name):
#    data_raw = request.data
#    data_raw = json.loads(data_raw)
#    error_code = data_raw['error_code']
#    module_name = data_raw['module_name'].upper()
#    template_name = data_raw['template_name']
#    regex = data_raw['regex'] if 'regex' in data_raw else None
#    json_description = data_raw['json_description'] if 'json_description' in data_raw else None
#    is_used = data_raw['is_used']
#
#    condition = {}
#    condition['regex'] = regex
#    condition['json_description'] = json_description
#
#    try:
#        utils2.update_center(module_name=module_name, table_name=table_name, template_name={0: template_name}, error_code=error_code, condition=condition, is_used=is_used)
#    except Exception as e:
#        print(e)
#
#    return "success"
#
#
#@app.route('/test_urlencode', methods=['POST', 'GET'])
#def test_urlencode():
#    data = request.data
#    data = parse.unquote(data.decode('utf-8'))
#    data = json.loads(data)
#    return ""
#
#
#if __name__ == '__main__':
#    app.run(
#      host='0.0.0.0',
#      port=5634,
#      debug=False
#    )
