import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def get_diff_from_ls(x):
    x.sort()
    return list(np.diff(x))


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
    register['r_day_diff'] = register.register_day.apply(lambda x: end_day - x)
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
        #         if day!=1:
        #             launch_temp['l_cnt_mean_in_'+str(day)] = launch_temp['l_cnt_sum_in_'+str(day)] / day
        launch = pd.merge(launch, launch_temp, on=['user_id'], how='left').fillna(0)
    launch['l_cnt_weight_sum'] = 0
    for day in [1, 3, 7]:
        launch['l_cnt_weight_sum'] += launch['l_cnt_sum_in_' + str(day)] * (8 - day)
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
    #     launch_temp['l_cnt_sum'] = launch_temp.cnt.apply(lambda x: sum(x))
    #     launch_temp['l_cnt_mean'] = launch_temp.cnt.apply(lambda x: np.mean(x))
    #     launch_temp['l_cnt_max'] = launch_temp.cnt.apply(lambda x: max(x))
    launch_temp['l_cnt_var'] = launch_temp.cnt.apply(lambda x: pd.Series(x).var())
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
    launch_temp['l_day_range'] = launch_temp.day.apply(lambda x: max(x) - min(x))
    launch_temp['l_day_std'] = launch_temp.day.apply(lambda x: pd.Series(x).std())
    launch = pd.merge(launch, launch_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return launch.fillna(-1)  # -1


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
        #         if day!=1:
        #             video_temp['v_cnt_mean_in_'+str(day)] = video_temp['v_cnt_sum_in_'+str(day)] / day
        video = pd.merge(video, video_temp, on=['user_id'], how='left').fillna(0)
    video['v_cnt_weight_sum'] = 0
    for day in [1, 3, 7]:
        video['v_cnt_weight_sum'] += video['v_cnt_sum_in_' + str(day)] * (8 - day)
    return video.fillna(0)


# 拍摄 日cnt 特征
def get_video_cnt_feature2(start_day, end_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    df = video_create_log_df[(video_create_log_df.day >= start_day) & (video_create_log_df.day <= end_day)][
        ['user_id', 'day']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'day']).agg('sum').reset_index()
    video_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    #     video_temp['v_cnt_count'] = video_temp.cnt.apply(lambda x: len(x))
    video_temp['v_cnt_sum'] = video_temp.cnt.apply(lambda x: sum(x))
    video_temp['v_cnt_mean'] = video_temp.cnt.apply(lambda x: np.mean(x))
    #     video_temp['v_cnt_max'] = video_temp.cnt.apply(lambda x: max(x))
    #     video_temp['v_cnt_var'] = video_temp.cnt.apply(lambda x: pd.Series(x).var())
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
    #     video_temp['v_day_range'] = video_temp.day.apply(lambda x: max(x)-min(x))
    video_temp['v_day_std'] = video_temp.day.apply(lambda x: pd.Series(x).std())
    video = pd.merge(video, video_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return video.fillna(-1)  # -1


# 拍摄 cnt差分 特征
def get_video_cnt_diff_feature(start_day, end_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    df = video_create_log_df[(video_create_log_df.day >= start_day) & (video_create_log_df.day <= end_day)][
        ['user_id', 'day']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'day']).agg('sum').reset_index()
    video_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    video_temp.cnt = video_temp.cnt.apply(lambda x: np.diff(x))
    #     video_temp['v_cnt_diff_max'] = video_temp.cnt.apply(lambda x: max(x) if len(x) != 0 else np.nan)
    video_temp['v_cnt_diff_min'] = video_temp.cnt.apply(lambda x: min(x) if len(x) != 0 else np.nan)
    video_temp['v_cnt_diff_mean'] = video_temp.cnt.apply(lambda x: np.mean(x) if len(x) != 0 else np.nan)
    video_temp['v_cnt_diff_std'] = video_temp.cnt.apply(lambda x: np.std(x) if len(x) != 0 else np.nan)
    #     video_temp['v_cnt_diff_skew'] = video_temp.cnt.apply(lambda x: pd.Series(x).skew())
    video_temp['v_cnt_diff_kurt'] = video_temp.cnt.apply(lambda x: pd.Series(x).kurt())
    video = pd.merge(video, video_temp.drop(['day', 'cnt'], axis=1), on=['user_id'], how='left')
    return video


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
        #         if day!=1:
        #             action_temp['a_cnt_mean_in_'+str(day)] = action_temp['a_cnt_sum_in_'+str(day)] / day
        action = pd.merge(action, action_temp, on=['user_id'], how='left').fillna(0)
    action['a_cnt_weight_sum'] = 0
    for day in [1, 3, 7]:
        action['a_cnt_weight_sum'] += action['a_cnt_sum_in_' + str(day)] * (8 - day)
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
    action_temp['a_cnt_var'] = action_temp.cnt.apply(lambda x: pd.Series(x).var())
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
    action_temp['a_day_range'] = action_temp.day.apply(lambda x: max(x) - min(x))
    action_temp['a_day_std'] = action_temp.day.apply(lambda x: pd.Series(x).std())
    action = pd.merge(action, action_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return action.fillna(-1)  # -1


# #行为 作者 特征
# def get_action_author_feature(start_day,end_day):
#     action = user_activity_log_df[['user_id']].drop_duplicates()
#     df = user_activity_log_df[(user_activity_log_df.day>=start_day)&(user_activity_log_df.day<=end_day)][['user_id','author_id']]
#     authors = set(df['author_id'])
#     action['is_author'] = action.user_id.apply(lambda x: 1 if x in authors else 0)
#     return action

# 行为 cnt差分 特征
def get_action_cnt_diff_feature(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day']]
    df['cnt'] = 1
    df = df.groupby(['user_id', 'day']).agg('sum').reset_index()
    action_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    action_temp.cnt = action_temp.cnt.apply(lambda x: np.diff(x))
    action_temp['a_cnt_diff_max'] = action_temp.cnt.apply(lambda x: max(x) if len(x) != 0 else np.nan)
    action_temp['a_cnt_diff_min'] = action_temp.cnt.apply(lambda x: min(x) if len(x) != 0 else np.nan)
    action_temp['a_cnt_diff_mean'] = action_temp.cnt.apply(lambda x: np.mean(x) if len(x) != 0 else np.nan)
    action_temp['a_cnt_diff_std'] = action_temp.cnt.apply(lambda x: np.std(x) if len(x) != 0 else np.nan)
    action_temp['a_cnt_diff_skew'] = action_temp.cnt.apply(lambda x: pd.Series(x).skew())
    action_temp['a_cnt_diff_kurt'] = action_temp.cnt.apply(lambda x: pd.Series(x).kurt())
    action = pd.merge(action, action_temp.drop(['day', 'cnt'], axis=1), on=['user_id'], how='left')
    return action  # -1


# 行为 视频和作者 特征
def get_action_video_author_feature(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day', 'video_id', 'author_id']]
    df.day = end_day - df.day
    for day in [1, 3, 7, end_day - start_day + 1]:
        # 看过的不同视频数量
        action_temp_df = df[df.day < day][['user_id', 'video_id']]
        action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: list(set(x))).reset_index()
        action_temp_df['a_video_set_cnt_in_' + str(day)] = action_temp_df.video_id.apply(lambda x: len(x))
        action_temp_df = action_temp_df.drop(['video_id'], axis=1)
        action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
        # 看过的不同作者数量
        action_temp_df = df[df.day < day][['user_id', 'author_id']]
        action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: list(set(x))).reset_index()
        action_temp_df['a_author_set_cnt_in_' + str(day)] = action_temp_df.author_id.apply(lambda x: len(x))
        action_temp_df = action_temp_df.drop(['author_id'], axis=1)
        action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    return action.fillna(0)


# 行为page cnt 特征
def get_action_page_feature1(start_day, end_day):
    pages = user_activity_log_df.page.unique()
    pages.sort()
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day', 'page']]
    df.day = end_day - df.day
    for page in pages:
        for day in [1, 3, 7, end_day - start_day + 1]:
            action_temp = df[(df.day < day) & (df.page == page)][['user_id']]
            action_temp['a_p' + str(page) + '_cnt_sum_in_' + str(day)] = 1
            action_temp = action_temp.groupby(['user_id']).agg('sum').reset_index()
            #             if (day!=1) and (day!=15):
            #                 action_temp['a_p'+str(page)+'_cnt_mean_in_'+str(day)] = action_temp['a_p'+str(page)+'_cnt_sum_in_'+str(day)] / day
            action = pd.merge(action, action_temp, on=['user_id'], how='left').fillna(0)
        action['a_p' + str(page) + '_cnt_weight_sum'] = 0
        for day in [1, 3, 7, end_day - start_day + 1]:
            action['a_p' + str(page) + '_cnt_weight_sum'] += action['a_p' + str(page) + '_cnt_sum_in_' + str(day)] * (
                end_day - start_day + 2 - day)
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
        for day in [1, 3, 7, end_day - start_day + 1]:
            action_temp = df[(df.day < day) & (df.action_type == atype)][['user_id']]
            action_temp['a_a' + str(atype) + '_cnt_sum_in_' + str(day)] = 1
            action_temp = action_temp.groupby(['user_id']).agg('sum').reset_index()
            #             if (day!=1) and (day!=15):
            #                 action_temp['a_a'+str(atype)+'_cnt_mean_in_'+str(day)] = action_temp['a_a'+str(atype)+'_cnt_sum_in_'+str(day)] / day
            action = pd.merge(action, action_temp, on=['user_id'], how='left').fillna(0)
        action['a_a' + str(atype) + '_cnt_weight_sum'] = 0
        for day in [1, 3, 7, end_day - start_day + 1]:
            action['a_a' + str(atype) + '_cnt_weight_sum'] += action['a_a' + str(atype) + '_cnt_sum_in_' + str(day)] * (
                end_day - start_day + 2 - day)
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


# 登录 日期差分 特征
def get_launch_day_diff_feature(start_day, end_day):
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    df = app_launch_log_df[(app_launch_log_df.day >= start_day) & (app_launch_log_df.day <= end_day)][
        ['user_id', 'day']]
    launch_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()  # 是否去重
    launch_temp.day = launch_temp.day.apply(lambda x: get_diff_from_ls(x))
    launch_temp['l_day_diff_max'] = launch_temp.day.apply(lambda x: max(x) if len(x) != 0 else np.nan)
    launch_temp['l_day_diff_min'] = launch_temp.day.apply(lambda x: min(x) if len(x) != 0 else np.nan)
    launch_temp['l_day_diff_mean'] = launch_temp.day.apply(lambda x: np.mean(x) if len(x) != 0 else np.nan)
    launch_temp['l_day_diff_std'] = launch_temp.day.apply(lambda x: np.std(x) if len(x) != 0 else np.nan)
    launch_temp['l_day_diff_skew'] = launch_temp.day.apply(lambda x: pd.Series(x).skew())
    launch_temp['l_day_diff_kurt'] = launch_temp.day.apply(lambda x: pd.Series(x).kurt())
    launch = pd.merge(launch, launch_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return launch  # -1


# 拍摄 日期差分 特征
def get_video_day_diff_feature(start_day, end_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    df = video_create_log_df[(video_create_log_df.day >= start_day) & (video_create_log_df.day <= end_day)][
        ['user_id', 'day']]
    video_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()  # 是否去重
    video_temp.day = video_temp.day.apply(lambda x: get_diff_from_ls(x))
    #     video_temp['v_day_diff_max'] = video_temp.day.apply(lambda x: max(x) if len(x) != 0 else np.nan)
    #     video_temp['v_day_diff_min'] = video_temp.day.apply(lambda x: min(x) if len(x) != 0 else np.nan)
    video_temp['v_day_diff_mean'] = video_temp.day.apply(lambda x: np.mean(x) if len(x) != 0 else np.nan)
    video_temp['v_day_diff_std'] = video_temp.day.apply(lambda x: np.std(x) if len(x) != 0 else np.nan)
    video_temp['v_day_diff_skew'] = video_temp.day.apply(lambda x: pd.Series(x).skew())
    video_temp['v_day_diff_kurt'] = video_temp.day.apply(lambda x: pd.Series(x).kurt())
    video = pd.merge(video, video_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return video  # -1


# 行为 日期差分 特征
def get_action_day_diff_feature(start_day, end_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    df = user_activity_log_df[(user_activity_log_df.day >= start_day) & (user_activity_log_df.day <= end_day)][
        ['user_id', 'day']]
    action_temp = df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()  # 是否去重
    action_temp.day = action_temp.day.apply(lambda x: get_diff_from_ls(x))
    action_temp['a_day_diff_max'] = action_temp.day.apply(lambda x: max(x) if len(x) != 0 else np.nan)
    #     action_temp['a_day_diff_min'] = action_temp.day.apply(lambda x: min(x) if len(x) != 0 else np.nan)
    action_temp['a_day_diff_mean'] = action_temp.day.apply(lambda x: np.mean(x) if len(x) != 0 else np.nan)
    action_temp['a_day_diff_std'] = action_temp.day.apply(lambda x: np.std(x) if len(x) != 0 else np.nan)
    action_temp['a_day_diff_skew'] = action_temp.day.apply(lambda x: pd.Series(x).skew())
    action_temp['a_day_diff_kurt'] = action_temp.day.apply(lambda x: pd.Series(x).kurt())
    action = pd.merge(action, action_temp.drop(['day'], axis=1), on=['user_id'], how='left')
    return action  # -1


# 留存特征
# 距离观测点(end_day)1天内、3天内和7天内留存率
def get_register_retention_feature1(start_day, end_day):
    register = user_register_log_df[user_register_log_df.register_day <= end_day][['user_id', 'register_day']]
    retention = register.groupby(['register_day']).agg({'user_id': 'count'}).reset_index()
    for day_cell in [3, 7]:
        app_temp_df = \
            app_launch_log_df[(app_launch_log_df.day <= end_day) & (app_launch_log_df.day > end_day - day_cell)][
                ['user_id']]
        video_temp_df = \
            video_create_log_df[(video_create_log_df.day <= end_day) & (video_create_log_df.day > end_day - day_cell)][
                ['user_id']]
        action_temp_df = \
            user_activity_log_df[
                (user_activity_log_df.day <= end_day) & (user_activity_log_df.day > end_day - day_cell)][
                ['user_id']]
        active_temp = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
        retention_temp = pd.merge(active_temp, register, on=['user_id'], how='left')
        retention_temp = retention_temp.groupby(['register_day']).agg('count').reset_index()
        retention_temp.rename(columns={'user_id': 'r_retention_in_' + str(day_cell)}, inplace=True)
        retention = pd.merge(retention, retention_temp, on=['register_day'], how='left')
        retention['r_retention_in_' + str(day_cell)] = retention['r_retention_in_' + str(day_cell)] / retention.user_id
    retention.rename(columns={'user_id': 'r_cnt_daily'}, inplace=True)
    return retention.fillna(0)


# 距离观测点(end_day)1天、3天和7天留存率
def get_register_retention_feature2(start_day, end_day):
    register = user_register_log_df[user_register_log_df.register_day <= end_day][['user_id', 'register_day']]
    retention = register.groupby(['register_day']).agg({'user_id': 'count'}).reset_index()
    for day_cell in [1, 3, 7]:
        app_temp_df = app_launch_log_df[app_launch_log_df.day == end_day - day_cell + 1][['user_id']]
        video_temp_df = video_create_log_df[video_create_log_df.day == end_day - day_cell + 1][['user_id']]
        action_temp_df = user_activity_log_df[user_activity_log_df.day == end_day - day_cell + 1][['user_id']]
        active_temp = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
        retention_temp = pd.merge(active_temp, register, on=['user_id'], how='left')
        retention_temp = retention_temp.groupby(['register_day']).agg('count').reset_index()
        retention_temp.rename(columns={'user_id': 'r_retention_at_' + str(day_cell)}, inplace=True)
        retention = pd.merge(retention, retention_temp, on=['register_day'], how='left')
        retention['r_retention_at_' + str(day_cell)] = retention['r_retention_at_' + str(day_cell)] / retention.user_id

    return retention.drop(['user_id'], axis=1).fillna(0)


# 注册日次日、3天、7天、观测点(end_day)留存率
def get_register_retention_feature3(start_day, end_day):
    register = user_register_log_df[user_register_log_df.register_day <= end_day][['user_id', 'register_day']]
    app_temp_df = app_launch_log_df[app_launch_log_df.day <= end_day][['user_id', 'day']]
    video_temp_df = video_create_log_df[video_create_log_df.day <= end_day][['user_id', 'day']]
    action_temp_df = user_activity_log_df[user_activity_log_df.day <= end_day][['user_id', 'day']]
    active_temp = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
    retention_temp = pd.merge(active_temp, register, on=['user_id'], how='left')
    retention_temp = retention_temp.groupby(['register_day', 'day']).agg('count').reset_index()
    retention_temp.rename(columns={'user_id': 'cnt'}, inplace=True)
    retention = register.groupby(['register_day']).agg({'user_id': 'count'}).reset_index()
    for day in [1, 3, 7, end_day]:
        if day != end_day:
            retention['r_' + str(day) + '_retention'] = retention.register_day.apply(
                lambda x: x + day if (x + day) < end_day else end_day)
            retention = pd.merge(retention, retention_temp, left_on=['register_day', 'r_' + str(day) + '_retention'],
                                 right_on=['register_day', 'day'], how='left')
            retention['r_' + str(day) + '_retention'] = retention.cnt / retention.user_id
            retention = retention.drop(['cnt', 'day'], axis=1)
        else:
            retention['r_endday_retention'] = retention.register_day.apply(
                lambda x: x + day if (x + day) < end_day else end_day)
            retention = pd.merge(retention, retention_temp, left_on=['register_day', 'r_endday_retention'],
                                 right_on=['register_day', 'day'], how='left')
            retention['r_endday_retention'] = retention.cnt / retention.user_id
            retention = retention.drop(['cnt', 'day'], axis=1)
    return retention.drop(['user_id'], axis=1).fillna(0)


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
    #     action4=get_action_author_feature(start_day,end_day)
    action5 = get_action_video_author_feature(start_day, end_day)

    # 行为page特征群
    page1 = get_action_page_feature1(start_day, end_day)
    page2 = get_action_page_feature2(start_day, end_day)

    # 行为actiontype特征群
    atype1 = get_action_atype_feature1(start_day, end_day)
    atype2 = get_action_atype_feature2(start_day, end_day)

    # cnt差分特征
    cnt_diff1 = get_video_cnt_diff_feature(start_day, end_day)
    cnt_diff2 = get_action_cnt_diff_feature(start_day, end_day)

    # day差分特征 不去重
    day_diff1 = get_launch_day_diff_feature(start_day, end_day)
    day_diff2 = get_video_day_diff_feature(start_day, end_day)
    day_diff3 = get_action_day_diff_feature(start_day, end_day)

    # 留存特征
    retention1 = get_register_retention_feature1(start_day, end_day)
    retention2 = get_register_retention_feature2(start_day, end_day)
    retention3 = get_register_retention_feature3(start_day, end_day)

    # 被行为特征
    #     actioned1=get_actioned_cnt_feature(start_day,end_day)

    data = pd.merge(register, launch1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch3, on=['user_id'], how='left').fillna(-1)

    data = pd.merge(data, video1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video3, on=['user_id'], how='left').fillna(-1)

    data = pd.merge(data, action1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action3, on=['user_id'], how='left').fillna(-1)
    #     data=pd.merge(data,action4,on=['user_id'],how='left').fillna(0)
    data = pd.merge(data, action5, on=['user_id'], how='left').fillna(0)

    #     data=pd.merge(data,actioned1,on=['user_id'],how='left').fillna(0)

    data = pd.merge(data, page1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, page2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, atype1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, atype2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, retention1, on=['register_day'], how='left').fillna(0)
    data = pd.merge(data, retention2, on=['register_day'], how='left').fillna(0)
    data = pd.merge(data, retention3, on=['register_day'], how='left').fillna(0)

    data = pd.merge(data, cnt_diff1, on=['user_id'], how='left')
    data = pd.merge(data, cnt_diff2, on=['user_id'], how='left')

    data = pd.merge(data, day_diff1, on=['user_id'], how='left')
    data = pd.merge(data, day_diff2, on=['user_id'], how='left')
    data = pd.merge(data, day_diff3, on=['user_id'], how='left')

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
    #     action4=get_action_author_feature(start_day,end_day)
    action5 = get_action_video_author_feature(start_day, end_day)

    # 行为page特征群
    page1 = get_action_page_feature1(start_day, end_day)
    page2 = get_action_page_feature2(start_day, end_day)

    # 行为actiontype特征群
    atype1 = get_action_atype_feature1(start_day, end_day)
    atype2 = get_action_atype_feature2(start_day, end_day)

    # cnt差分特征
    cnt_diff1 = get_video_cnt_diff_feature(start_day, end_day)
    cnt_diff2 = get_action_cnt_diff_feature(start_day, end_day)

    # day差分特征 不去重
    day_diff1 = get_launch_day_diff_feature(start_day, end_day)
    day_diff2 = get_video_day_diff_feature(start_day, end_day)
    day_diff3 = get_action_day_diff_feature(start_day, end_day)

    # 留存特征
    retention1 = get_register_retention_feature1(start_day, end_day)
    retention2 = get_register_retention_feature2(start_day, end_day)
    retention3 = get_register_retention_feature3(start_day, end_day)

    # 被行为特征
    #     actioned1=get_actioned_cnt_feature(start_day,end_day)

    data = pd.merge(register, launch1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, launch3, on=['user_id'], how='left').fillna(-1)

    data = pd.merge(data, video1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, video3, on=['user_id'], how='left').fillna(-1)

    data = pd.merge(data, action1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action2, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, action3, on=['user_id'], how='left').fillna(-1)
    #     data=pd.merge(data,action4,on=['user_id'],how='left').fillna(0)
    data = pd.merge(data, action5, on=['user_id'], how='left').fillna(0)

    #     data=pd.merge(data,actioned1,on=['user_id'],how='left').fillna(0)

    data = pd.merge(data, page1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, page2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, atype1, on=['user_id'], how='left').fillna(0)
    data = pd.merge(data, atype2, on=['user_id'], how='left').fillna(0)

    data = pd.merge(data, retention1, on=['register_day'], how='left').fillna(0)
    data = pd.merge(data, retention2, on=['register_day'], how='left').fillna(0)
    data = pd.merge(data, retention3, on=['register_day'], how='left').fillna(0)

    data = pd.merge(data, cnt_diff1, on=['user_id'], how='left')
    data = pd.merge(data, cnt_diff2, on=['user_id'], how='left')

    data = pd.merge(data, day_diff1, on=['user_id'], how='left')
    data = pd.merge(data, day_diff2, on=['user_id'], how='left')
    data = pd.merge(data, day_diff3, on=['user_id'], how='left')

    return data


if __name__ == '__main__':
    INPUT_BASE_PATH = ''
    OUTPUT_BASE_PATH = ''

    # 加载数据
    user_register_log_df, app_launch_log_df, video_create_log_df, user_activity_log_df = load_data(INPUT_BASE_PATH)

    # 训练集
    starttime = datetime.datetime.now()
    train = get_train_data(1, 16)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,data1 done,shape:' + str(
        train.shape) + '==========')

    # 验证集
    starttime = datetime.datetime.now()
    valid = get_train_data(8, 23)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,data2 done,shape:' + str(
        valid.shape) + '==========')

    # 合并data1和data2
    starttime = datetime.datetime.now()
    df = pd.concat([train, valid], axis=0)
    df = shuffle(df)
    drop_columns = ['a_a4_cnt_sum_in_1', 'a_a4_cnt_sum_in_16', 'a_a4_cnt_sum_in_3', 'a_a4_cnt_sum_in_7',
                    'a_a4_cnt_weight_sum', 'a_a4_weight', 'a_a5_cnt_sum_in_1', 'a_a5_cnt_sum_in_3', 'l_cnt_var']
    df = df.drop(drop_columns, axis=1)
    df.to_csv(OUTPUT_BASE_PATH + '/train.csv', index=False)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,train done,shape:' + str(
        test.shape) + '==========')

    # 测试集
    starttime = datetime.datetime.now()
    test = get_test_data(15, 30)
    test.to_csv(OUTPUT_BASE_PATH + '/test.csv', index=False)
    print('========' + str((datetime.datetime.now() - starttime).seconds) + 's,data3 done,shape:' + str(
        test.shape) + '==========')
