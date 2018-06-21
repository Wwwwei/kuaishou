import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def load_data(base_path):
    user_register_log_df = pd.read_csv(base_path + '/user_register_log.txt', sep='\t', header=None,
                                       names=['user_id', 'register_day', 'register_type', 'device_type'])
    app_launch_log_df = pd.read_csv(base_path + '/app_launch_log.txt', sep='\t', header=None, names=['user_id', 'day'])
    video_create_log_df = pd.read_csv(base_path + '/video_create_log.txt', sep='\t', header=None,
                                      names=['user_id', 'day'])
    user_activity_log_df = pd.read_csv(base_path + '/user_activity_log.txt', sep='\t', header=None,
                                       names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'])
    return user_register_log_df, app_launch_log_df, video_create_log_df, user_activity_log_df


# 注册特征
def get_register_feature(start_day, end_day):
    register = user_register_log_df[user_register_log_df.register_day <= end_day].copy()
    register['register_day_diff'] = register.register_day.apply(lambda x: end_day - x)
    return register


# 获取label
def get_label(start_day, end_day):
    app_temp_df = app_launch_log_df[(app_launch_log_df.day <= end_day) & (app_launch_log_df.day >= start_day)][
        ['user_id']].drop_duplicates()
    video_temp_df = video_create_log_df[(video_create_log_df.day <= end_day) & (video_create_log_df.day >= start_day)][
        ['user_id']].drop_duplicates()
    action_temp_df = \
        user_activity_log_df[(user_activity_log_df.day <= end_day) & (user_activity_log_df.day >= start_day)][
            ['user_id']].drop_duplicates()
    label = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
    label['label'] = 1
    return label


# 登录 cnt 特征
def get_launch_cnt_feature1(start_day, end_day):
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    df = app_launch_log_df[(app_launch_log_df.day >= start_day) & (app_launch_log_df.day <= end_day)][
        ['user_id', 'day']]
    df.day = end_day - df.day
    for day in [1, 3, 7]:
        launch_temp = df[df.day < day][['user_id']]
        launch_temp['l_cnt_sum_in_' + str(day)] = 1
        launch_temp = launch_temp.groupby(['user_id']).agg('sum').reset_index()
        if day != 1:
            launch_temp['l_cnt_mean_in_' + str(day)] = launch_temp['l_cnt_sum_in_' + str(day)] / day
        launch = pd.merge(launch, launch_temp, on=['user_id'], how='left')
    return launch.fillna(0)


# 登录 日cnt 特征(用户每天只登陆一次 无意义)
def get_launch_cnt_feature2(start_day, end_day):
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    df = app_launch_log_df[(app_launch_log_df.day >= start_day) & (app_launch_log_df.day <= end_day)][
        ['user_id', 'day']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'day']).agg('sum').reset_index()
    launch_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    launch_temp['l_cnt_count'] = launch_temp.cnt.apply(lambda x: len(x))
    launch_temp['l_cnt_sum'] = launch_temp.cnt.apply(lambda x: sum(x))
    launch_temp['l_cnt_mean'] = launch_temp.cnt.apply(lambda x: np.mean(x))
    launch_temp['l_cnt_max'] = launch_temp.cnt.apply(lambda x: max(x))
    launch_temp['l_cnt_var'] = launch_temp.cnt.apply(lambda x: np.var(x))
    launch = pd.merge(launch, launch_temp.drop(['day', 'cnt'], axis=1), on=['user_id'], how='left')
    return launch.fillna(0)


# 登录 日期 特征
def get_launch_day_feature1(start_day, end_day):
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    df = app_launch_log_df[(app_launch_log_df.day >= start_day) & (app_launch_log_df.day <= end_day)][
        ['user_id', 'day']]
    df.day = end_day - df.day
    launch_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    launch_temp['l_day_max'] = launch_temp.day.apply(lambda x: max(x))
    launch_temp['l_day_min'] = launch_temp.day.apply(lambda x: min(x))
    launch_temp['l_day_std'] = launch_temp.day.apply(lambda x: np.std(x))
    launch = pd.merge(launch, launch_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return launch.fillna(0)  # -1


# 拍摄 cnt 特征
def get_video_cnt_feature1(start_day, end_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    df = video_create_log_df[(video_create_log_df.day >= start_day) & (video_create_log_df.day <= end_day)][
        ['user_id', 'day']]
    df.day = end_day - df.day
    for day in [1, 3, 7]:
        video_temp = df[df.day < day][['user_id']]
        video_temp['v_cnt_sum_in_' + str(day)] = 1
        video_temp = video_temp.groupby(['user_id']).agg('sum').reset_index()
        if day != 1:
            video_temp['v_cnt_mean_in_' + str(day)] = video_temp['v_cnt_sum_in_' + str(day)] / day
        video = pd.merge(video, video_temp, on=['user_id'], how='left')
    return video.fillna(0)


# 拍摄 日cnt 特征
def get_video_cnt_feature2(start_day, end_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    df = video_create_log_df[(video_create_log_df.day >= start_day) & (video_create_log_df.day <= end_day)][
        ['user_id', 'day']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'day']).agg('sum').reset_index()
    video_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    video_temp['v_cnt_count'] = video_temp.cnt.apply(lambda x: len(x))
    video_temp['v_cnt_sum'] = video_temp.cnt.apply(lambda x: sum(x))
    video_temp['v_cnt_mean'] = video_temp.cnt.apply(lambda x: np.mean(x))
    video_temp['v_cnt_max'] = video_temp.cnt.apply(lambda x: max(x))
    video_temp['v_cnt_var'] = video_temp.cnt.apply(lambda x: np.var(x))
    video = pd.merge(video, video_temp.drop(['day', 'cnt'], axis=1), on=['user_id'], how='left')
    return video.fillna(0)


# 拍摄 日期 特征
def get_video_day_feature1(start_day, end_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    df = video_create_log_df[(video_create_log_df.day >= start_day) & (video_create_log_df.day <= end_day)][
        ['user_id', 'day']]
    df.day = end_day - df.day
    video_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    video_temp['v_day_max'] = video_temp.day.apply(lambda x: max(x))
    video_temp['v_day_min'] = video_temp.day.apply(lambda x: min(x))
    video_temp['v_day_std'] = video_temp.day.apply(lambda x: np.std(x))
    video = pd.merge(video, video_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return video.fillna(0)  # -1


# 行为 cnt 特征
def get_action_cnt_feature1(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day']]
    df.day = end_day - df.day
    for day in [1, 3, 7]:
        action_temp = df[df.day < day][['user_id']]
        action_temp['a_cnt_sum_in_' + str(day)] = 1
        action_temp = action_temp.groupby(['user_id']).agg('sum').reset_index()
        if day != 1:
            action_temp['a_cnt_mean_in_' + str(day)] = action_temp['a_cnt_sum_in_' + str(day)] / day
        action = pd.merge(action, action_temp, on=['user_id'], how='left')
    return action.fillna(0)


# 行为 日cnt 特征
def get_action_cnt_feature2(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'day']).agg('sum').reset_index()
    action_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    action_temp['a_cnt_count'] = action_temp.cnt.apply(lambda x: len(x))
    action_temp['a_cnt_sum'] = action_temp.cnt.apply(lambda x: sum(x))
    action_temp['a_cnt_mean'] = action_temp.cnt.apply(lambda x: np.mean(x))
    action_temp['a_cnt_max'] = action_temp.cnt.apply(lambda x: max(x))
    action_temp['a_cnt_var'] = action_temp.cnt.apply(lambda x: np.var(x))
    action = pd.merge(action, action_temp.drop(['day', 'cnt'], axis=1), on=['user_id'], how='left')
    return action.fillna(0)


# 行为 日期 特征
def get_action_day_feature1(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day']]
    df.day = end_day - df.day
    action_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    action_temp['a_day_max'] = action_temp.day.apply(lambda x: max(x))
    action_temp['a_day_min'] = action_temp.day.apply(lambda x: min(x))
    action_temp['a_day_std'] = action_temp.day.apply(lambda x: np.std(x))
    action = pd.merge(action, action_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return action.fillna(0)  # -1


# 行为page cnt 特征
def get_action_page_feature1(start_day, end_day):
    pages = user_activity_log_df.page.unique()
    pages.sort()
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day', 'page']]
    df.day = end_day - df.day
    for page in pages:
        for day in [1, 3, 7]:
            action_temp = df[(df.day < day) & (df.page == page)][['user_id']]
            action_temp['a_p' + str(page) + '_cnt_sum_in_' + str(day)] = 1
            action_temp = action_temp.groupby(['user_id']).agg('sum').reset_index()
            if day != 1:
                action_temp['a_p' + str(page) + '_cnt_mean_in_' + str(day)] = action_temp['a_p' + str(
                    page) + '_cnt_sum_in_' + str(day)] / day
            action = pd.merge(action, action_temp, on=['user_id'], how='left')
    return action.fillna(0)


# 行为page 占比 特征
def get_action_page_feature2(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day', 'page']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'page']).agg(lambda x: x.shape[0]).unstack().reset_index().fillna(0)
    action_temp = pd.DataFrame()
    action_temp['user_id'] = df['user_id']
    df['page_cnt'] = df.cnt.apply(lambda x: x.sum(), axis=1)
    action_temp['a_p0_weight'] = df.cnt[0] / df.page_cnt
    action_temp['a_p1_weight'] = df.cnt[1] / df.page_cnt
    action_temp['a_p2_weight'] = df.cnt[2] / df.page_cnt
    action_temp['a_p3_weight'] = df.cnt[3] / df.page_cnt
    action_temp['a_p4_weight'] = df.cnt[4] / df.page_cnt
    action = pd.merge(action, action_temp, on=['user_id'], how='left')
    return action.fillna(0)


# 行为type cnt 特征
def get_action_atype_feature1(start_day, end_day):
    atypes = user_activity_log_df.action_type.unique()
    atypes.sort()
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day', 'action_type']]
    df.day = end_day - df.day
    for atype in atypes:
        for day in [1, 3, 7]:
            action_temp = df[(df.day < day) & (df.action_type == atype)][['user_id']]
            action_temp['a_a' + str(atype) + '_cnt_sum_in_' + str(day)] = 1
            action_temp = action_temp.groupby(['user_id']).agg('sum').reset_index()
            if day != 1:
                action_temp['a_a' + str(atype) + '_cnt_mean_in_' + str(day)] = action_temp['a_a' + str(
                    atype) + '_cnt_sum_in_' + str(day)] / day
            action = pd.merge(action, action_temp, on=['user_id'], how='left')
    return action.fillna(0)


# 行为type 占比 特征
def get_action_atype_feature2(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day', 'action_type']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'action_type']).agg(lambda x: x.shape[0]).unstack().reset_index().fillna(0)
    action_temp = pd.DataFrame()
    action_temp['user_id'] = df['user_id']
    df['atype_cnt'] = df.cnt.apply(lambda x: x.sum(), axis=1)
    action_temp['a_a0_weight'] = df.cnt[0] / df.atype_cnt
    action_temp['a_a1_weight'] = df.cnt[1] / df.atype_cnt
    action_temp['a_a2_weight'] = df.cnt[2] / df.atype_cnt
    action_temp['a_a3_weight'] = df.cnt[3] / df.atype_cnt
    action_temp['a_a4_weight'] = df.cnt[4] / df.atype_cnt
    action_temp['a_a5_weight'] = df.cnt[5] / df.atype_cnt
    action = pd.merge(action, action_temp, on=['user_id'], how='left')
    return action.fillna(0)


def get_train_data(start_day, end_day):
    # 注册特征群
    register = get_register_feature(start_day, end_day)

    # 标签
    label = get_label(end_day + 1, end_day + 7)

    # 启动特征群
    launch1 = get_launch_cnt_feature1(start_day, end_day)
    launch2 = get_launch_cnt_feature2(start_day, end_day)
    launch3 = get_launch_day_feature1(start_day, end_day)

    # 拍摄特征群
    video1 = get_video_cnt_feature1(start_day, end_day)
    video2 = get_video_cnt_feature2(start_day, end_day)
    video3 = get_video_day_feature1(start_day, end_day)

    # 行为特征群
    action1 = get_action_cnt_feature1(start_day, end_day)
    action2 = get_action_cnt_feature2(start_day, end_day)
    action3 = get_action_day_feature1(start_day, end_day)

    # 行为page特征群
    page1 = get_action_page_feature1(start_day, end_day)
    page2 = get_action_page_feature2(start_day, end_day)

    # 行为actiontype特征群
    atype1 = get_action_atype_feature1(start_day, end_day)
    atype2 = get_action_atype_feature2(start_day, end_day)

    data = pd.merge(register, launch1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch3, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, video1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video3, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, action1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action3, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, page1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, page2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, atype1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, atype2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, label, on=['user_id'], how='left')
    data.label = data.label.apply(lambda x: x if x == 1 else 0)

    return data


def get_test_data(start_day, end_day):
    # 注册特征群
    register = get_register_feature(start_day, end_day)

    # 启动特征群
    launch1 = get_launch_cnt_feature1(start_day, end_day)
    launch2 = get_launch_cnt_feature2(start_day, end_day)
    launch3 = get_launch_day_feature1(start_day, end_day)

    # 拍摄特征群
    video1 = get_video_cnt_feature1(start_day, end_day)
    video2 = get_video_cnt_feature2(start_day, end_day)
    video3 = get_video_day_feature1(start_day, end_day)

    # 行为特征群
    action1 = get_action_cnt_feature1(start_day, end_day)
    action2 = get_action_cnt_feature2(start_day, end_day)
    action3 = get_action_day_feature1(start_day, end_day)

    # 行为page特征群
    page1 = get_action_page_feature1(start_day, end_day)
    page2 = get_action_page_feature2(start_day, end_day)

    # 行为actiontype特征群
    atype1 = get_action_atype_feature1(start_day, end_day)
    atype2 = get_action_atype_feature2(start_day, end_day)

    data = pd.merge(register, launch1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch3, on=['user_id'], how='left').fillna(0)  # -1

    data = pd.merge(data, video1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video3, on=['user_id'], how='left').fillna(0)  # -1

    data = pd.merge(data, action1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action3, on=['user_id'], how='left').fillna(0)  # -1

    data = pd.merge(data, page1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, page2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, atype1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, atype2, on=['user_id'], how='left').fillna(0)

    return data


if __name__ == '__main__':
    INPUT_BASE_PATH = '/Users/zhaosw/Desktop/kuaishou/data'
    OUTPUT_BASE_PATH = '/Users/zhaosw/Desktop/kuaishou/data2'

    # 加载数据
    user_register_log_df, app_launch_log_df, video_create_log_df, user_activity_log_df = load_data(INPUT_BASE_PATH)

    # 训练集
    starttime = datetime.datetime.now()
    train = get_train_data(1, 15)
    train.to_csv(OUTPUT_BASE_PATH + '/data1.csv', index=False)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,data1 done,shape:' + str(
        train.shape) + '==========')

    # 验证集
    starttime = datetime.datetime.now()
    valid = get_train_data(9, 23)
    valid.to_csv(OUTPUT_BASE_PATH + '/data2.csv', index=False)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,data2 done,shape:' + str(
        valid.shape) + '==========')

    # 验证集
    starttime = datetime.datetime.now()
    test = get_test_data(16, 30)
    test.to_csv(OUTPUT_BASE_PATH + '/data3.csv', index=False)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,data3 done,shape:' + str(
        test.shape) + '==========')
