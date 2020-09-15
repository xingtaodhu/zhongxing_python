import datetime
import random
import datetime

import pymysql

# 取出数据库中的全部数据
def extract(cursor, db, module_name="all", table_name="mod_template"):
    mod_dict = {}
    value_list = ['error_code', 'date', 'module_name', 'template', 'regex', 'json_description', 'used']
    sql_all = f"select  error_code, date, module_name, template, regex, json_description, used from {table_name} where deleted=False"
    try:
        cursor.execute(sql_all)
        results = cursor.fetchall()
        # 获取全部分类信息
        for idx, item in enumerate(results):
            unit_dict = {}
            for dx, key in enumerate(item):
                unit_dict[value_list[dx]] = key

            if unit_dict["module_name"] not in mod_dict:
                mod_dict[unit_dict["module_name"]] = []
            mod_dict[unit_dict["module_name"]].append(unit_dict)

        return mod_dict
    except Exception as e:
        print(e)
        return {}



# 取出数据库中的template分类
def template_extract(cursor, db, module_name="CMS", table_name="mod_template", is_train="train", is_used=True):

    mod_dict = []
    value_list = ['error_code', 'date', 'module_name', 'template', 'regex', 'json_description', 'used']
    sql_pre_del = f"delete from {table_name} where module_name=%s and deleted=True"
    sql_pre_new = f"UPDATE {table_name} set waiting=False where module_name=%s and waiting=True"

    if is_used == None:
        sql = f"select  error_code, date, module_name, template, regex, json_description, used from {table_name} where module_name=%s"
    else:
        if is_train == "predict":
            sql = f"select  error_code, date, module_name, template, regex, json_description, used from {table_name} where module_name=%s and waiting=False and used=%s"
        elif is_train == "train":
            sql = f"select  error_code, date, module_name, template, regex, json_description, used from {table_name} where module_name=%s and used=%s"
        else:
            sql = f"select  error_code, date, module_name, template, regex, json_description, used from {table_name} where module_name=%s and deleted=False and used=%s"

    try:
        if is_train == "train":
            # 如果要进行训练，则先删除deleted为1的值，再把waiting为1的值变为0，表示加入等待区的模块
            cursor.execute(sql_pre_del, [module_name])
            cursor.execute(sql_pre_new, [module_name])
            db.commit()
        if is_used == None:
            cursor.execute(sql, [module_name])
        else:
            cursor.execute(sql, [module_name, is_used])

        results = cursor.fetchall()
        # 获取全部分类信息
        for idx, item in enumerate(results):
            unit_dict = {}
            for dx, key in enumerate(item):
                unit_dict[value_list[dx]] = key
            mod_dict.append(unit_dict)
        return mod_dict

    except Exception as e:
        print(e)
        return {}


# 存储抽取的新模板
def template_save(cursor, db, module_name="CMS", table_name="mod_template", template_name={}, is_backup=True, condition=None):
    # sql_backup = f"INSERT INTO {table_name} (module_name, error_code, template, regex, json_description) VALUES (%s, %s, %s, %s, %s)"
    sql = f"INSERT INTO {table_name} (module_name, error_code, template, regex, json_description, deleted, used, date, waiting) VALUES (%s, %s, %s, %s, %s, False, %s, %s, %s)"

    length = len(template_name.items())
    seed = max(length, 1000000)
    rand_id = random.sample(range(1, seed), length)
    i = 0
    if is_backup:
        is_used = False
        waiting = False
    else:
        is_used = True
        waiting = True

    if condition is None:
        try:
            for item in template_name.items():
                if len(item[1]) > 100:
                    item_modified = item[1][:100]
                else:
                    item_modified = item[1]

                # 设计序号
                curr_time = datetime.datetime.now()
                time_str = curr_time.strftime("%Y%m%d%H%M%S")
                new_id = time_str + str(rand_id[i]).zfill(8)
                i += 1


                # if is_backup:
                #     cursor.execute(sql_backup, [module_name, new_id, item_modified, None, None])
                # else:
                #     cursor.execute(sql, [module_name, new_id, item_modified, None, None])
                cursor.execute(sql, [module_name, new_id, item_modified, None, None, is_used, datetime.datetime.now(), waiting])

            # 获取全部分类信息
            db.commit()
        except Exception as e:
            print(e)

    else:
        try:
            for item in template_name.items():
                if len(item[1]) > 100:
                    item_modified = item[1][:100]
                else:
                    item_modified = item[1]

                curr_time = datetime.datetime.now()
                time_str = curr_time.strftime("%Y%m%d%H%M%S")
                new_id = time_str + str(rand_id[i]).zfill(8)
                i += 1

################2
                # if is_backup:
                #     cursor.execute(sql_backup, [module_name, new_id, item_modified, condition['regex'], condition['json_description']])
                # else:
                #     cursor.execute(sql, [module_name, new_id, item_modified, condition['regex'], condition['json_description']])
                cursor.execute(sql, [module_name, new_id, item_modified, condition['regex'], condition['json_description'], is_used, datetime.datetime.now(), waiting])

            db.commit()

        except Exception as e:
            print(e)


# 删除数据库中模板
def template_delete(cursor, db, error_code=None, module_name="CMS", table_name="mod_template", is_used=False):



    if error_code is None:
        sql = f"delete from {table_name} where module_name=%s and used=%s"
        try:
            cursor.execute(sql, [module_name, is_used])
            db.commit()
        except Exception as e:
            print(e)
    else:
        if is_used:
            sql_pre = f"SELECT waiting FROM {table_name} WHERE error_code=%s"
            try:
                cursor.execute(sql_pre, [error_code])
                results = cursor.fetchall()

                # waiting为1，即待加入模块直接删除
                if results[0][0] == 1:
                    sql = f"delete from {table_name} where error_code=%s"
                else:
                    sql = f"UPDATE {table_name} set deleted=True where error_code=%s"

            except Exception as e:
                print(e)

        else:
            sql = f"delete from {table_name} where error_code=%s"

        # sql = f"delete from {table_name} where error_code=%s"

        try:
            cursor.execute(sql, [error_code])
            db.commit()
        except Exception as e:
            print(e)


# 更新数据库中的模块
def template_update(cursor, db, module_name="CMS", table_name="mod_template", template_name={}, error_code=None, condition=None, is_used=False):

    # 制作新序号
    rand_id = random.sample(range(1, 1000000), 1)
    curr_time = datetime.datetime.now()
    time_str = curr_time.strftime("%Y%m%d%H%M%S")
    new_id = time_str + str(rand_id[0]).zfill(8)

    if is_used:
        sql_pre = f"SELECT waiting FROM {table_name} WHERE error_code=%s"
        try:
            cursor.execute(sql_pre, [error_code])
            results = cursor.fetchall()

            # 如果准备修改的不是waiting为0，即是处于待加入取得模板则需要生成一个新的待加入模板，然后将旧模板设deleted设为1
            if results[0][0] == 0:
                sql_copy = f"SELECT module_name, error_code, regex, json_description, deleted, used, date, waiting FROM {table_name} WHERE error_code=%s"
                cursor.execute(sql_copy, [error_code])
                results = cursor.fetchall()

                sql_new = f"INSERT INTO {table_name} (module_name, error_code, template, regex, json_description, deleted, used, date, waiting) VALUES (%s, %s, %s, %s, %s, False, %s, %s, True)"
                for item in template_name.items():
                    if len(item[1]) > 100:
                        item_modified = item[1][:100]
                    else:
                        item_modified = item[1]
                    cursor.execute(sql_new, [results[0][0], new_id, item_modified, condition['regex'], condition['json_description'], is_used, datetime.datetime.now()])

                sql_del = f"UPDATE {table_name} set deleted=True where error_code=%s"
                cursor.execute(sql_del, [error_code])
                db.commit()

            else:
                sql = f"UPDATE {table_name} set template=%s, regex=%s, json_description=%s, used=%s, date=%s where error_code=%s"
                for item in template_name.items():
                    if len(item[1]) > 100:
                        item_modified = item[1][:100]
                    else:
                        item_modified = item[1]

                    cursor.execute(sql, [item_modified, condition['regex'], condition['json_description'], is_used,
                                         datetime.datetime.now(), error_code])

                db.commit()

        except Exception as e:
            print(e)


    else:
        sql = f"UPDATE {table_name} set template=%s, regex=%s, json_description=%s, used=%s, date=%s where error_code=%s"

        try:
            for item in template_name.items():
                if len(item[1]) > 100:
                    item_modified = item[1][:100]
                else:
                    item_modified = item[1]

                cursor.execute(sql, [item_modified, condition['regex'], condition['json_description'], is_used, datetime.datetime.now(), error_code])

            db.commit()

        except Exception as e:
            print(e)
