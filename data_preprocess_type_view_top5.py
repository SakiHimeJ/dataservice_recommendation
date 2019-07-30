import pandas as pd
import numpy as np
import time
import math
import datetime

def get_dataset():
    res = pd.read_csv('./data/test_type_view_top5_id_tn_raw.csv', encoding='GBK')

    drop_list = []
    i = 0
    for r in res['ifview']:
        if np.isnan(r):
            drop_list.append(i)
        i += 1
    res = res.drop(drop_list, axis=0)



    res['label'] = res['ifview']
    res = res.drop(['ifview', 'ifsuggest'], axis=1)
    res = res.sample(frac=1)  # 打乱顺序
    res = res.reset_index(drop=True)

    '''
    # 总有几行异常的手动剔除
    i = 0
    drop_list = []
    for r in res['buy_eval_good']:
        if type(r) is str and len(r) > 8:
            # print(r)
            # print(i)
            drop_list.append(i)
        i += 1

    i = 0
    for r in res['mem_reg_time']:
        if len(r) < 14:
            # print(r)
            # print(i)
            drop_list.append(i)
        i += 1'''

    res = res.drop(drop_list, axis=0)

    res = res.reset_index(drop=True)

    # print(res.info())

    res['buy_eval_good'] = res['buy_eval_good'].astype(float)
    # print(res.info())

    res['birthday'] = res['birthday'].apply(lambda x: None if isinstance(x, str) else x)

    ffill_cols = ['is_daiguan', 'approve_time', 'first_shelf_time', 'license_day', 'is_local', 'gearbox_type', 'gps', 'is_limit_mileage', 'is_accept_get_return',
                  'gps_service_charge', 'ly_car_param_source', 'status', 'reply_flag', 'car_msg_communication_check_status', 'car_voice_communication_check_status',
                  'car_cate', 'owner_type', 'cc_unit', 'car_city', 'city_type', 'brand_txt', 'type_txt', 'model_txt', 'car_year', 'month', 'district']

    for col in res.columns:
        if col in ffill_cols:
            res[col] = res[col].fillna(method='ffill', axis=0)

    meanfill_cols = ['birthday', 'total_rent', 'buy_eval_good', 'buy_times', 'buy_res_avg_time', 'car_age', 'cpi', 'cylinder_capacity', 'real_cylinder_capacity',
                     'guide_price', 'guide_day_price', 'hour_price', 'day_price', 'week_price', 'holiday_price', 'day_mileage', 'get_car_lon', 'get_car_lat',
                     'req_count', 'req_count_uv', 'settle_trans_count', 'halfyear_settletrans_count', '1year_settletrans_count', 'successtrans_count', 'halfyear_successtrans_count',
                    '1year_successtrans_count', 'break_count', 'success_rent_day', 'halfyear_success_rentday', '1year_success_rentday', 'refuse_count',  '1year_refuse_count', 'halfyear_refuse_count',
                    'owner_cancel_count', '1year_owner_cancel_count', 'halfyear_owner_cancel_count', 'renter_cancel_count', '1year_renter_cancel_count', 'halfyear_renter_cancel_count', 'total_show_num',
                    '90day_show_num', 'total_favorit_num', '90day_favorit_num', 'total_click_num', 'total_click_num_all',  'total_click_num_uv_all',  '90day_click_num','placeorder_num', 'placeorder_num_uv']


    for col in res.columns:
        if col in meanfill_cols:
            res[col] = res[col].fillna(value=res[col].mean())



    res['dri_lic_st_time'] = res['dri_lic_st_time'].fillna(value='2015-01-23 21:47:25')
    res['mem_reg_time'] = res['mem_reg_time'].fillna(value='2014-09-05 11:41:49')
    res['mem_dri_lic_first_time'] = res['mem_dri_lic_first_time'].fillna(value='2011-10-11')
    res['start_validity_period'] = res['start_validity_period'].fillna(value='2000-06-20 00:00:00')



    # 时间格式转换
    meta_year = time.strptime("1970-1-1 00:00:00", "%Y-%m-%d %H:%M:%S")
    res['mem_reg_time'] = res['mem_reg_time'].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    res['dri_lic_st_time'] = res['dri_lic_st_time'].apply(lambda x:  time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    res['mem_dri_lic_first_time'] = res['mem_dri_lic_first_time'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S') if len(x) > 10 else time.strptime(x, '%Y-%m-%d'))
    res['mem_dri_lic_first_time'] = res['mem_dri_lic_first_time'].apply(lambda x: time.mktime(x) if x > meta_year else time.mktime(time.strptime('1970-1-2', '%Y-%m-%d')))
    res['start_validity_period'] = res['start_validity_period'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S') if len(x) > 10 else time.strptime(x, '%Y-%m-%d'))
    res['start_validity_period'] = res['start_validity_period'].apply(lambda x: time.mktime(x) if x > meta_year else time.mktime(time.strptime('1970-1-2', '%Y-%m-%d')))

    # print(res.info())

    # 离散特征数值化

    # discrete_cols = ['car_no', 'mem_no', 'dri_lic_allow_car', 'cc_unit']
    discrete_cols = ['car_no', 'typeid', 'mem_no', 'owner_type', 'car_city', 'city_type', 'brand_txt', 'type_txt', 'model_txt', 'cc_unit',
                     'car_year','month','district','car_cate','is_local','gearbox_type', 'is_limit_mileage', 'gps_service_charge', 'status']
    for col in res.columns:
        if col in discrete_cols:
            res[col] = pd.factorize(res[col].values, sort=True)[0]

    # min-max 归一化

    continuous_cols = []  # 暂时不做，在后一个函数里做归一化
    for column in continuous_cols:
        res[column] = (res[column] - res[column].min()) / (res[column].max() - res[column].min())


    # print(pos)
    # print(neg.values.shape)

    print(res.info(null_counts=True))
    # res.to_csv('./data/test_type_view_top5_id_tn.csv', encoding='GBK', index=False)

    return res


def get_fm_data():
    train_dict = {'index': [], 'value': [], 'label': [],  'feature_sizes':[]}
    val_dict = {'index': [], 'value': [], 'label': []}
    test_dict = {'index': [], 'value': [], 'label': []}

    df = pd.read_csv('./data/test_type_view_top5_id_tn.csv', encoding='GBK')
    df = df.sample(frac=1)  # 打乱顺序
    df = df.reset_index(drop=True)

    # feature engineering res
    # del_cols = ['mem_no', 'car_no']
    del_cols = ['mem_no', 'typeid', 'is_limit_mileage''', 'is_accept_get_return', 'is_daiguan', 'gps', 'gearbox_type',
                'reply_flag', 'successtrans_count', 'gps_service_charge',
                'car_msg_communication_check_status', 'halfyear_successtrans_count', 'halfyear_owner_cancel_count',
                'break_count', 'car_voice_communication_check_status',
                'status', 'real_cylinder_capacity', 'car_cate', 'owner_cancel_count', 'day_mileage', 'cc_unit',
                'halfyear_refuse_count', 'is_local', 'city_type', '1year_owner_cancel_count',
                'car_year', '1year_settletrans_count', '1year_refuse_count', 'ly_car_param_source',
                '1year_successtrans_count', 'license_day', 'halfyear_renter_cancel_count', '1year_renter_cancel_count',
                'week_price', 'halfyear_settletrans_count', 'refuse_count', 'settle_trans_count', 'owner_type',
                'req_count', 'renter_cancel_count']

    for col in df.columns:
        if col in del_cols:
            df = df.drop([col], axis=1)

    print(df.info(null_counts=True))

    df_cols = list(df.columns.values)
    discrete_cols_df = ['car_no', 'typeid', 'mem_no', 'owner_type', 'car_city', 'city_type', 'brand_txt', 'type_txt', 'model_txt',
                     'cc_unit',
                     'car_year', 'month', 'district', 'car_cate', 'is_local', 'gearbox_type', 'is_limit_mileage',
                     'gps_service_charge', 'status']


    discrete_cols = []
    i = 0
    for col in df_cols:
        if col in discrete_cols_df:
            discrete_cols.append(i)
        i += 1

    x = df.iloc[:, :-1].values
    x_index = x.copy()
    x_value = x.copy()
    y = df['label'].values.astype(int)


    for col in range(x.shape[1]):
        if col in discrete_cols:
            train_dict['feature_sizes'].append(len(np.unique(x[:, col])))
            x_index[:, col] = x_index[:, col].astype(int)
            x_value[:, col] = 1
        else:
            train_dict['feature_sizes'].append(1)
            x_index[:, col] = 0
            x_value[:, col] = (x_value[:, col] - x_value[:, col].min()) / (x_value[:, col].max() - x_value[:, col].min())
            x_value[:, col] = x_value[:, col].astype(float)

    # 7:2:1划分训练：验证：测试
    div_ind_val = math.floor(x.shape[0] * (1 - 0.3))
    div_ind_test = math.floor(x.shape[0] * (1 - 0.1))

    train_dict['index'] = x_index[:div_ind_val].tolist()
    train_dict['value'] = x_value[:div_ind_val].tolist()
    train_dict['label'] = y[:div_ind_val].tolist()

    val_dict['index'] = x_index[div_ind_val:div_ind_test].tolist()
    val_dict['value'] = x_value[div_ind_val:div_ind_test].tolist()
    val_dict['label'] = y[div_ind_val:div_ind_test].tolist()

    test_dict['index'] = x_index[div_ind_test:].tolist()
    test_dict['value'] = x_value[div_ind_test:].tolist()
    test_dict['label'] = y[div_ind_test:].tolist()


    return train_dict, val_dict, test_dict


if __name__ == "__main__":
    # get_dataset()
    # get_fm_data()
    pass
