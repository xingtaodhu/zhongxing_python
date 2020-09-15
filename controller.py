import datetime
import random

import pymysql


# 取出数据库中的template分类
def template_extract(cursor, db, module_name="CMS", table_name="mod_template", is_train=False, id_need=False, is_used=True):

    sql_pre = f"delete from {table_name} where module_name=%s and deleted=True"
    sql = f"select error_code, template from {table_name} where module_name=%s and used=%s"
    try:
        if is_train:
            cursor.execute(sql_pre, [module_name])
        cursor.execute(sql, [module_name, is_used])
        results = cursor.fetchall()
        # 获取全部分类信息
        if id_need:
            mod_dict = {row[0]: row[1] for idx, row in enumerate(results)}
        else:
            mod_dict = {idx: row[1] for idx, row in enumerate(results)}
        return mod_dict
    except Exception as e:
        print(e)
        return {}


# 存储抽取的新模板
def template_save(cursor, db, module_name="CMS", table_name="mod_template", template_name={}, is_backup=True, condition=None):

    # sql_backup = f"INSERT INTO {table_name} (module_name, error_code, template, regex, json_description) VALUES (%s, %s, %s, %s, %s)"
    sql = f"INSERT INTO {table_name} (module_name, error_code, template, regex, json_description, deleted, used) VALUES (%s, %s, %s, %s, %s, False, %s)"

    length = len(template_name.items())
    seed = max(length, 1000000)
    rand_id = random.sample(range(1, seed), length)
    i = 0
    if is_backup:
        is_used = False
    else:
        is_used = True

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
                cursor.execute(sql, [module_name, new_id, item_modified, None, None, is_used])

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
                cursor.execute(sql, [module_name, new_id, item_modified, condition['regex'], condition['json_description'], is_used])

            db.commit()

        except Exception as e:
            print(e)


# 删除数据库中模板
def template_delete(cursor, db, error_code=None, module_name="CMS", table_name="mod_template", is_used=False):
    if error_code is None:
        sql = f"delete from {table_name} where module_name=%s and used={is_used}"
        try:
            cursor.execute(sql, [module_name, is_used])
            db.commit()
        except Exception as e:
            print(e)
    else:
        sql = f"delete from {table_name} where error_code=%s and used={is_used}"

        try:
            cursor.execute(sql, [error_code])
            db.commit()
        except Exception as e:
            print(e)


# 更新数据库中的模块
def template_update(cursor, db, module_name="CMS", table_name="mod_template", template_name={}, error_code=None, condition=None, is_used=False):

    sql = f"UPDATE {table_name} set template=%s, regex=%s, json_description=%s, used=%s where error_code=%s"

    try:
        for item in template_name.items():
            if len(item[1]) > 100:
                item_modified = item[1][:100]
            else:
                item_modified = item[1]

            cursor.execute(sql, [item_modified, condition['regex'], condition['json_description'], is_used, error_code])

        db.commit()

    except Exception as e:
        print(e)
