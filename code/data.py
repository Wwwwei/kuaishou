import pandas as pd
import numpy as np
import datetime

day_cells = [1, 2, 3, 5, 7, 10, 16]  # 对于窗口内的数据，统计的区间为连续1，3，5，7，10，15天（其中15天为全量）


# 工具函数
def get_max_serial_days_from_list(x):
    x = list(set(x))
    ls = []
    for i in range(0, len(x)):
        days = 1
        for j in range(i, len(x) - 1):
            if x[j + 1] == (x[j] + 1):
                days = days + 1
            else:
                break
        ls.append(days)
    return max(ls)


def get_diff_from_ls(x):
    x.sort()
    return list(np.diff(x))


# load data
def load_data(base_path):
    user_register_log_df = pd.read_csv(base_path + '/user_register_log.txt', sep='\t', header=None,
                                       names=['user_id', 'register_day', 'register_type', 'device_type'])
    app_launch_log_df = pd.read_csv(base_path + '/app_launch_log.txt', sep='\t', header=None,
                                    names=['user_id', 'day'])
    video_create_log_df = pd.read_csv(base_path + '/video_create_log.txt', sep='\t', header=None,
                                      names=['user_id', 'day'])
    user_activity_log_df = pd.read_csv(base_path + '/user_activity_log.txt', sep='\t', header=None,
                                       names=['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type'])
    return user_register_log_df, app_launch_log_df, video_create_log_df, user_activity_log_df


user_register_log_df, app_launch_log_df, video_create_log_df, user_activity_log_df = load_data(
    r"F:\downloads_Python\Test\kesci\data\raw_data\\")


# =============================================================================
# 启动日志特征
# =============================================================================
# 登录 cnt 特征  (距窗口节点1,3,5,7,10,15天的，登录天数次数，3天内次数均值)
def get_launch_feature1(current_day):
    # 启动特征时间聚集
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    for day_cell in day_cells:
        app_temp_df = \
            app_launch_log_df[
                (app_launch_log_df.day < current_day) & (app_launch_log_df.day >= current_day - day_cell)][
                ['user_id']]
        app_temp_df['launch_cnt_in_' + str(day_cell) + '_days'] = 1
        app_temp_df = app_temp_df.groupby(['user_id']).agg('sum').reset_index()
        # TODO: 目前是逻辑平均
        if day_cell != 1:
            app_temp_df['launch_cnt_mean_in_' + str(day_cell) + '_days'] = app_temp_df['launch_cnt_in_' + str(
                day_cell) + '_days'] / day_cell
        launch = pd.merge(launch, app_temp_df, on=['user_id'], how='left')
    launch = launch.fillna(0)
    return launch


# 登录 日期(current_day-day) 特征
def get_launch_feature2(current_day):
    app_launch_log_temp_df = app_launch_log_df[
        (app_launch_log_df.day < current_day) & (app_launch_log_df.day >= current_day - day_cells[-1])]
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    app_temp_df = app_launch_log_temp_df[['user_id', 'day']]
    app_temp_df = app_temp_df.groupby(['user_id']).aggregate(lambda x: list(current_day - x)).reset_index()
    app_temp_df['launch_day_mean'] = app_temp_df.day.apply(lambda x: np.mean(x))
    app_temp_df['launch_day_var'] = app_temp_df.day.apply(lambda x: np.var(x))
    app_temp_df['launch_day_std'] = app_temp_df.day.apply(lambda x: np.std(x))
    app_temp_df['launch_day_median'] = app_temp_df.day.apply(lambda x: np.median(x))
    app_temp_df['launch_day_mode'] = app_temp_df.day.apply(lambda x: np.argmax(np.bincount(x)))
    app_temp_df.day = app_temp_df.day.apply(lambda x: list(set(x)))
    app_temp_df['launch_day_max'] = app_temp_df.day.apply(lambda x: max(x))
    app_temp_df['launch_day_min'] = app_temp_df.day.apply(lambda x: min(x))
    app_temp_df['launch_day_interval'] = app_temp_df.day.apply(lambda x: max(x) - min(x))
    app_temp_df['launch_day_set_cnt'] = app_temp_df.day.apply(lambda x: len(x))

    app_temp_df = app_temp_df.drop(['day'], axis=1)
    launch = pd.merge(launch, app_temp_df, on=['user_id'], how='left')
    launch = launch.fillna(-1)
    return launch


# 登录 日期间隔 特征
def get_launch_feature3(current_day):
    app_launch_log_temp_df = app_launch_log_df[
        (app_launch_log_df.day < current_day) & (app_launch_log_df.day >= current_day - day_cells[-1])]
    launch = app_launch_log_df[['user_id']].drop_duplicates()
    app_temp_df = app_launch_log_temp_df[['user_id', 'day']]
    app_temp_df = app_temp_df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    app_temp_df['launch_serial_max'] = app_temp_df.day.apply(lambda x: get_max_serial_days_from_list(x))

    app_temp_df.day = app_temp_df.day.apply(lambda x: get_diff_from_ls(x))
    app_temp_df['launch_interval_min'] = app_temp_df.day.apply(lambda x: min(x) if len(x) != 0 else -1)
    app_temp_df['launch_interval_max'] = app_temp_df.day.apply(lambda x: max(x) if len(x) != 0 else -1)
    app_temp_df['launch_interval_mean'] = app_temp_df.day.apply(lambda x: np.mean(x) if len(x) != 0 else -1)
    app_temp_df['launch_interval_var'] = app_temp_df.day.apply(lambda x: np.var(x) if len(x) != 0 else -1)
    app_temp_df['launch_interval_std'] = app_temp_df.day.apply(lambda x: np.std(x) if len(x) != 0 else -1)

    app_temp_df = app_temp_df.drop(['day'], axis=1)
    launch = pd.merge(launch, app_temp_df, on=['user_id'], how='left')
    launch = launch.fillna(-1)
    return launch


# =============================================================================
# 拍摄日志特征
# =============================================================================
# 拍摄 cnt 特征
def get_video_feature1(current_day):
    video = video_create_log_df[['user_id']].drop_duplicates()
    for day_cell in day_cells:
        video_temp_df = video_create_log_df[
            (video_create_log_df.day < current_day) & (video_create_log_df.day >= current_day - day_cell)][['user_id']]
        video_temp_df['video_cnt_in_' + str(day_cell) + '_days'] = 1
        video_temp_df = video_temp_df.groupby(['user_id']).agg('sum').reset_index()
        # TODO: 目前是逻辑平均
        if day_cell != 1:
            video_temp_df['video_cnt_mean_in_' + str(day_cell) + '_days'] = video_temp_df['video_cnt_in_' + str(
                day_cell) + '_days'] / day_cell
        video = pd.merge(video, video_temp_df, on=['user_id'], how='left')
    video = video.fillna(0)
    return video


# 拍摄 日期(current_day-day) 特征
def get_video_feature2(current_day):
    video_create_log_temp_df = video_create_log_df[
        (video_create_log_df.day < current_day) & (video_create_log_df.day >= current_day - day_cells[-1])]
    video = video_create_log_df[['user_id']].drop_duplicates()

    video_temp_df = video_create_log_temp_df[['user_id', 'day']]
    video_temp_df = video_temp_df.groupby(['user_id']).aggregate(lambda x: list(current_day - x)).reset_index()
    video_temp_df['video_day_mean'] = video_temp_df.day.apply(lambda x: np.mean(x))
    video_temp_df['video_day_var'] = video_temp_df.day.apply(lambda x: np.var(x))
    video_temp_df['video_day_std'] = video_temp_df.day.apply(lambda x: np.std(x))
    video_temp_df['video_day_median'] = video_temp_df.day.apply(lambda x: np.median(x))
    video_temp_df['video_day_mode'] = video_temp_df.day.apply(lambda x: np.argmax(np.bincount(x)))

    video_temp_df.day = video_temp_df.day.apply(lambda x: list(set(x)))
    video_temp_df['video_day_max'] = video_temp_df.day.apply(lambda x: max(x))
    video_temp_df['video_day_min'] = video_temp_df.day.apply(lambda x: min(x))
    video_temp_df['video_day_interval'] = video_temp_df.day.apply(lambda x: max(x) - min(x))
    video_temp_df['video_day_set_cnt'] = video_temp_df.day.apply(lambda x: len(x))

    video_temp_df = video_temp_df.drop(['day'], axis=1)
    video = pd.merge(video, video_temp_df, on=['user_id'], how='left')
    video = video.fillna(-1)
    return video


# 拍摄 日期间隔 特征
def get_video_feature3(current_day):
    video_create_log_temp_df = video_create_log_df[
        (video_create_log_df.day < current_day) & (video_create_log_df.day >= current_day - day_cells[-1])]
    video = video_create_log_df[['user_id']].drop_duplicates()

    video_temp_df = video_create_log_temp_df[['user_id', 'day']]
    video_temp_df = video_temp_df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    video_temp_df['video_serial_max'] = video_temp_df.day.apply(lambda x: get_max_serial_days_from_list(x))

    video_temp_df.day = video_temp_df.day.apply(lambda x: get_diff_from_ls(x))
    video_temp_df['video_interval_min'] = video_temp_df.day.apply(lambda x: min(x) if len(x) != 0 else -1)
    video_temp_df['video_interval_max'] = video_temp_df.day.apply(lambda x: max(x) if len(x) != 0 else -1)
    video_temp_df['video_interval_mean'] = video_temp_df.day.apply(lambda x: np.mean(x) if len(x) != 0 else -1)
    video_temp_df['video_interval_var'] = video_temp_df.day.apply(lambda x: np.var(x) if len(x) != 0 else -1)
    video_temp_df['video_interval_std'] = video_temp_df.day.apply(lambda x: np.std(x) if len(x) != 0 else -1)

    video_temp_df = video_temp_df.drop(['day'], axis=1)
    video = pd.merge(video, video_temp_df, on=['user_id'], how='left')
    video = video.fillna(-1)
    return video


# 拍摄 日cnt 特征
def get_video_feature4(current_day):
    video_create_log_temp_df = video_create_log_df[
        (video_create_log_df.day < current_day) & (video_create_log_df.day >= current_day - day_cells[-1])]
    video = video_create_log_df[['user_id']].drop_duplicates()
    video_temp_df = video_create_log_temp_df[['user_id', 'day']]
    video_temp_df['cnt'] = 1
    video_temp_df = video_temp_df.groupby(['user_id', 'day']).sum().reset_index()
    video_temp_df = video_temp_df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
    video_temp_df['video_daily_cnt_min'] = video_temp_df.cnt.apply(lambda x: min(x))
    video_temp_df['video_daily_cnt_max'] = video_temp_df.cnt.apply(lambda x: max(x))
    video_temp_df['video_daily_cnt_mean'] = video_temp_df.day.apply(lambda x: np.mean(x))
    video_temp_df = video_temp_df.drop(['day', 'cnt'], axis=1)
    video = pd.merge(video, video_temp_df, on=['user_id'], how='left')
    video = video.fillna(0)
    return video


# =============================================================================
# 行为日志特征
# =============================================================================
# #行为 cnt 特征
def get_action_feature1(current_day):
    action = user_activity_log_df[['user_id']].drop_duplicates()
    for day_cell in day_cells:
        action_temp_df = user_activity_log_df[
            (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cell)][
            ['user_id']]
        action_temp_df['action_cnt_in_' + str(day_cell) + '_days'] = 1
        action_temp_df = action_temp_df.groupby(['user_id']).agg('sum').reset_index()
        # TODO: 目前是逻辑平均
        if day_cell != 1:
            action_temp_df['action_cnt_mean_in_' + str(day_cell) + '_days'] = action_temp_df['action_cnt_in_' + str(
                day_cell) + '_days'] / day_cell
        action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# 行为 日期(current_day-day) 特征
def get_action_feature2(current_day):
    user_activity_log_temp_df = user_activity_log_df[
        (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cells[-1])]
    action = user_activity_log_df[['user_id']].drop_duplicates()

    action_temp_df = user_activity_log_temp_df[['user_id', 'day']]
    action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: list(current_day - x)).reset_index()
    action_temp_df['action_day_mean'] = action_temp_df.day.apply(lambda x: np.mean(x))
    action_temp_df['action_day_var'] = action_temp_df.day.apply(lambda x: np.var(x))
    action_temp_df['action_day_std'] = action_temp_df.day.apply(lambda x: np.std(x))
    action_temp_df['action_day_median'] = action_temp_df.day.apply(lambda x: np.median(x))
    action_temp_df['action_day_mode'] = action_temp_df.day.apply(lambda x: np.argmax(np.bincount(x)))

    action_temp_df.day = action_temp_df.day.apply(lambda x: list(set(x)))
    action_temp_df['action_day_max'] = action_temp_df.day.apply(lambda x: max(x))
    action_temp_df['action_day_min'] = action_temp_df.day.apply(lambda x: min(x))
    action_temp_df['action_day_interval'] = action_temp_df.day.apply(lambda x: max(x) - min(x))
    action_temp_df['action_day_set_cnt'] = action_temp_df.day.apply(lambda x: len(x))

    action_temp_df = action_temp_df.drop(['day'], axis=1)
    action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    action = action.fillna(-1)
    return action


# 行为 视频和作者 特征
def get_action_feature3(current_day):
    action_f1 = user_activity_log_df[['user_id']].drop_duplicates()
    action_f2 = user_activity_log_df[['user_id']].drop_duplicates()

    for day_cell in day_cells:
        user_activity_log_temp_df = user_activity_log_df[
            (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cell)]
        # 行为的video数量时间聚集特征
        action_temp_df = user_activity_log_temp_df[['user_id', 'video_id']]
        action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: set(x)).reset_index()
        action_temp_df['action_video_num_in_' + str(day_cell) + '_days'] = action_temp_df.video_id.apply(
            lambda x: len(x))
        action_temp_df['action_video_set_num_in_' + str(day_cell) + '_days'] = action_temp_df.video_id.apply(
            lambda x: len(set(x)))
        action_temp_df = action_temp_df.drop(['video_id'], axis=1)
        action_f1 = pd.merge(action_f1, action_temp_df, on=['user_id'], how='left')
        # 行为的用户数量时间聚集特征
        action_temp_df = user_activity_log_temp_df[['user_id', 'author_id']]
        action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: set(x)).reset_index()
        action_temp_df['action_author_num_in_' + str(day_cell) + '_days'] = action_temp_df.author_id.apply(
            lambda x: len(x))
        action_temp_df['action_author_set_num_in_' + str(day_cell) + '_days'] = action_temp_df.author_id.apply(
            lambda x: len(set(x)))
        action_temp_df = action_temp_df.drop(['author_id'], axis=1)
        action_f2 = pd.merge(action_f2, action_temp_df, on=['user_id'], how='left')

    action = pd.merge(action_f1, action_f2, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# 行为page 占比 特征
def get_action_page_feature1(current_day):
    pages = user_activity_log_df.page.unique()
    action = user_activity_log_df[['user_id']].drop_duplicates()

    action_temp_df = user_activity_log_df[
        (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cells[-1])][
        ['user_id', 'page']]
    action_temp_df['cnt'] = 1
    action_temp_df = action_temp_df.groupby(['user_id', 'page']).agg(
        lambda x: x.shape[0]).unstack().reset_index().fillna(0)
    action_temp = pd.DataFrame()
    action_temp['user_id'] = action_temp_df['user_id']
    action_temp_df['page_cnt'] = action_temp_df.cnt.apply(lambda x: x.sum(), axis=1)
    action_temp['page0_weight'] = action_temp_df.cnt[0] / action_temp_df.page_cnt
    action_temp['page1_weight'] = action_temp_df.cnt[1] / action_temp_df.page_cnt
    action_temp['page2_weight'] = action_temp_df.cnt[2] / action_temp_df.page_cnt
    action_temp['page3_weight'] = action_temp_df.cnt[3] / action_temp_df.page_cnt
    action_temp['page4_weight'] = action_temp_df.cnt[4] / action_temp_df.page_cnt
    action_temp['page_cnt'] = action_temp_df['page_cnt']

    action = pd.merge(action, action_temp, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# 行为page cnt 特征
def get_action_page_feature2(current_day):
    pages = user_activity_log_df.page.unique()
    action = user_activity_log_df[['user_id']].drop_duplicates()

    for day_cell in day_cells:
        user_activity_log_temp_df = user_activity_log_df[
            (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cell)]
        for page in pages:
            action_temp_df = user_activity_log_temp_df[user_activity_log_temp_df.page == page][['user_id']].copy()
            action_temp_df['action_page' + str(page) + '_cnt_in_' + str(day_cell) + '_days'] = 1
            action_temp_df = action_temp_df.groupby(['user_id']).agg('sum').reset_index()
            # TODO: 目前是逻辑平均
            if day_cell != 1:
                action_temp_df['action_page' + str(page) + '_cnt_mean_in_' + str(day_cell) + '_days'] = action_temp_df[
                                                                                                            'action_page' + str(
                                                                                                                page) + '_cnt_in_' + str(
                                                                                                                day_cell) + '_days'] / day_cell
            action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# 行为actiontype 占比 特征
def get_action_actiontype_feature1(current_day):
    pages = user_activity_log_df.page.unique()
    action = user_activity_log_df[['user_id']].drop_duplicates()

    action_temp_df = user_activity_log_df[
        (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cells[-1])][
        ['user_id', 'action_type']]
    action_temp_df['cnt'] = 1
    action_temp_df = action_temp_df.groupby(['user_id', 'action_type']).agg(
        lambda x: x.shape[0]).unstack().reset_index().fillna(0)
    action_temp = pd.DataFrame()
    action_temp['user_id'] = action_temp_df['user_id']
    action_temp_df['action_cnt'] = action_temp_df.cnt.apply(lambda x: x.sum(), axis=1)
    action_temp['action0_weight'] = action_temp_df.cnt[0] / action_temp_df.action_cnt
    action_temp['action1_weight'] = action_temp_df.cnt[1] / action_temp_df.action_cnt
    action_temp['action2_weight'] = action_temp_df.cnt[2] / action_temp_df.action_cnt
    action_temp['action3_weight'] = action_temp_df.cnt[3] / action_temp_df.action_cnt
    action_temp['action4_weight'] = action_temp_df.cnt[4] / action_temp_df.action_cnt
    action_temp['action5_weight'] = action_temp_df.cnt[5] / action_temp_df.action_cnt
    action_temp['action_cnt'] = action_temp_df['action_cnt']

    action = pd.merge(action, action_temp, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# 行为actiontype cnt 特征
def get_action_actiontype_feature2(current_day):
    action_types = user_activity_log_df.action_type.unique()
    action = user_activity_log_df[['user_id']].drop_duplicates()

    for day_cell in day_cells:
        user_activity_log_temp_df = user_activity_log_df[
            (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cell)]
        for action_type in action_types:
            action_temp_df = user_activity_log_temp_df[user_activity_log_temp_df.action_type == action_type][
                ['user_id']].copy()
            action_temp_df['action' + str(action_type) + '_cnt_in_' + str(day_cell) + '_days'] = 1
            action_temp_df = action_temp_df.groupby(['user_id']).agg('sum').reset_index()
            # TODO: 目前是逻辑平均
            if day_cell != 1:
                action_temp_df['action' + str(action_type) + '_cnt_mean_in_' + str(day_cell) + '_days'] = \
                    action_temp_df['action' + str(action_type) + '_cnt_in_' + str(day_cell) + '_days'] / day_cell
            action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# 行为actiontype 日期(current_day-day) 特征
def get_action_actiontype_feature3(current_day):
    action_types = user_activity_log_df.action_type.unique()
    user_activity_log_temp_df = user_activity_log_df[
        (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cells[-1])][
        ['user_id', 'day', 'action_type']]
    action = user_activity_log_df[['user_id']].drop_duplicates()
    for action_type in action_types:
        action_temp_df = user_activity_log_temp_df[user_activity_log_temp_df.action_type == action_type][
            ['user_id', 'day']]
        action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: list(current_day - x)).reset_index()
        action_temp_df['action' + str(action_type) + '_day_mean'] = action_temp_df.day.apply(lambda x: np.mean(x))
        action_temp_df['action' + str(action_type) + '_day_var'] = action_temp_df.day.apply(lambda x: np.var(x))
        action_temp_df['action' + str(action_type) + '_day_std'] = action_temp_df.day.apply(lambda x: np.std(x))
        action_temp_df['action' + str(action_type) + '_day_median'] = action_temp_df.day.apply(lambda x: np.median(x))
        action_temp_df['action' + str(action_type) + '_day_mode'] = action_temp_df.day.apply(
            lambda x: np.argmax(np.bincount(x)))

        action_temp_df.day = action_temp_df.day.apply(lambda x: list(set(x)))
        action_temp_df['action' + str(action_type) + '_day_max'] = action_temp_df.day.apply(lambda x: max(x))
        action_temp_df['action' + str(action_type) + '_day_min'] = action_temp_df.day.apply(lambda x: min(x))
        action_temp_df['action' + str(action_type) + '_day_interval'] = action_temp_df.day.apply(
            lambda x: max(x) - min(x))
        action_temp_df['action' + str(action_type) + '_day_set_cnt'] = action_temp_df.day.apply(lambda x: len(x))
        action_temp_df = action_temp_df.drop(['day'], axis=1)
        action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    action = action.fillna(-1)

    return action


# 行为actiontype 日cnt 特征
def get_action_actiontype_feature4(current_day):
    action_types = user_activity_log_df.action_type.unique()
    user_activity_log_temp_df = user_activity_log_df[
        (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cells[-1])][
        ['user_id', 'day', 'action_type']]
    action = user_activity_log_df[['user_id']].drop_duplicates()
    for action_type in action_types:
        action_temp_df = user_activity_log_temp_df[user_activity_log_temp_df.action_type == action_type][
            ['user_id', 'day']]
        action_temp_df['cnt'] = 1
        action_temp_df = action_temp_df.groupby(['user_id', 'day']).sum().reset_index()
        action_temp_df = action_temp_df.groupby(['user_id']).aggregate(lambda x: list(x)).reset_index()
        action_temp_df['action' + str(action_type) + '_daily_cnt_min'] = action_temp_df.cnt.apply(lambda x: min(x))
        action_temp_df['action' + str(action_type) + '_daily_cnt_max'] = action_temp_df.cnt.apply(lambda x: max(x))
        action_temp_df['action' + str(action_type) + '_daily_cnt_mean'] = action_temp_df.day.apply(lambda x: np.mean(x))
        action_temp_df['action' + str(action_type) + '_daily_cnt_var'] = action_temp_df.day.apply(lambda x: np.var(x))
        action_temp_df = action_temp_df.drop(['day', 'cnt'], axis=1)
        action = pd.merge(action, action_temp_df, on=['user_id'], how='left')
    action = action.fillna(0)
    return action


# =============================================================================
# 获取label
def get_label(current_day):
    app_temp_df = app_launch_log_df[(app_launch_log_df.day < current_day + 7) & (app_launch_log_df.day >= current_day)][
        ['user_id']].drop_duplicates()
    video_temp_df = \
        video_create_log_df[(video_create_log_df.day < current_day + 7) & (video_create_log_df.day >= current_day)][
            ['user_id']].drop_duplicates()
    action_temp_df = \
        user_activity_log_df[(user_activity_log_df.day < current_day + 7) & (user_activity_log_df.day >= current_day)][
            ['user_id']].drop_duplicates()
    active = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
    active['label'] = 1
    return active


# 注册特征
def get_register_feature(current_day):
    register = user_register_log_df[user_register_log_df.register_day < current_day].copy()
    register['register_day_diff'] = register.register_day.apply(lambda x: current_day - x)
    return register


# 活跃特征
def get_active_feature(current_day):
    active = get_register_feature(current_day)[['user_id']].drop_duplicates()
    for day_cell in day_cells:
        app_temp_df = \
            app_launch_log_df[
                (app_launch_log_df.day < current_day) & (app_launch_log_df.day >= current_day - day_cell)][
                ['user_id']]
        video_temp_df = video_create_log_df[
            (video_create_log_df.day < current_day) & (video_create_log_df.day >= current_day - day_cell)][['user_id']]
        action_temp_df = user_activity_log_df[
            (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cell)][
            ['user_id']]
        active_temp = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0)
        active_temp['active_cnt_in_' + str(day_cell) + '_days'] = 1
        active_temp = active_temp.groupby(['user_id']).agg('sum').reset_index()
        active = pd.merge(active, active_temp, on=['user_id'], how='left')
    active = active.fillna(0)
    return active


# 留存特征
def get_retention_feature1(current_day):
    register = user_register_log_df[user_register_log_df.register_day < current_day][['user_id', 'register_day']]
    retention = register.groupby(['register_day']).agg({'user_id': 'count'}).reset_index()
    for day_cell in day_cells:
        app_temp_df = \
            app_launch_log_df[
                (app_launch_log_df.day < current_day) & (app_launch_log_df.day >= current_day - day_cell)][
                ['user_id']]
        video_temp_df = video_create_log_df[
            (video_create_log_df.day < current_day) & (video_create_log_df.day >= current_day - day_cell)][['user_id']]
        action_temp_df = user_activity_log_df[
            (user_activity_log_df.day < current_day) & (user_activity_log_df.day >= current_day - day_cell)][
            ['user_id']]
        active_temp = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
        retention_temp = pd.merge(active_temp, register, on=['user_id'], how='left')
        retention_temp = retention_temp.groupby(['register_day']).agg('count').reset_index()
        retention_temp.rename(columns={'user_id': 'retention_rate_in_' + str(day_cell) + '_days'}, inplace=True)
        retention = pd.merge(retention, retention_temp, on=['register_day'], how='left')
        retention['retention_rate_in_' + str(day_cell) + '_days'] = retention['retention_rate_in_' + str(
            day_cell) + '_days'] / retention.user_id
    retention = retention.fillna(0)
    retention.rename(columns={'user_id': 'register_cnt'}, inplace=True)
    return retention


def get_retention_feature2(current_day):
    register = user_register_log_df[user_register_log_df.register_day < current_day][['user_id', 'register_day']]
    retention = register.groupby(['register_day']).agg({'user_id': 'count'}).reset_index()
    for day_cell in day_cells:
        app_temp_df = app_launch_log_df[(app_launch_log_df.day >= current_day - day_cells[-1]) & (
            app_launch_log_df.day < current_day - day_cells[-1] + day_cell)][['user_id']]
        video_temp_df = video_create_log_df[(video_create_log_df.day >= current_day - day_cells[-1]) & (
            video_create_log_df.day < current_day - day_cells[-1] + day_cell)][['user_id']]
        action_temp_df = user_activity_log_df[(user_activity_log_df.day >= current_day - day_cells[-1]) & (
            user_activity_log_df.day < current_day - day_cells[-1] + day_cell)][['user_id']]
        active_temp = pd.concat([app_temp_df, video_temp_df, action_temp_df], axis=0).drop_duplicates()
        retention_temp = pd.merge(active_temp, register, on=['user_id'], how='left')
        retention_temp = retention_temp.groupby(['register_day']).agg('count').reset_index()
        retention_temp.rename(columns={'user_id': 'retention_rate_after_' + str(day_cell) + '_days'}, inplace=True)
        retention = pd.merge(retention, retention_temp, on=['register_day'], how='left')
        retention['retention_rate_after_' + str(day_cell) + '_days'] = retention['retention_rate_after_' + str(
            day_cell) + '_days'] / retention.user_id
    retention = retention.fillna(0)
    retention = retention.drop(['user_id'], axis=1)
    return retention


# 特征提取
def get_train_data(current_day):
    # 启动特征群
    launch1 = get_launch_feature1(current_day)
    launch2 = get_launch_feature2(current_day)
    launch3 = get_launch_feature3(current_day)
    # 拍摄特征群
    video1 = get_video_feature1(current_day)
    video2 = get_video_feature2(current_day)
    video3 = get_video_feature3(current_day)
    video4 = get_video_feature4(current_day)
    # 行为特征群
    action1 = get_action_feature1(current_day)
    action2 = get_action_feature2(current_day)
    action3 = get_action_feature3(current_day)
    # 行为page特征群
    page1 = get_action_page_feature1(current_day)
    page2 = get_action_page_feature2(current_day)
    # 行为actiontype特征群
    actiontype1 = get_action_actiontype_feature1(current_day)
    actiontype2 = get_action_actiontype_feature2(current_day)
    actiontype3 = get_action_actiontype_feature3(current_day)
    actiontype4 = get_action_actiontype_feature4(current_day)

    # 注册特征群
    register = get_register_feature(current_day)
    # 活跃特征群
    active = get_active_feature(current_day)
    # 留存率
    retention1 = get_retention_feature1(current_day)
    retention2 = get_retention_feature2(current_day)
    # 标签
    label = get_label(current_day)

    data = pd.merge(register, launch1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, launch2, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, launch3, on=['user_id'], how='left')
    data = data.fillna(-1)

    data = pd.merge(data, video1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, video2, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, video3, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, video4, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, action1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, action2, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, action3, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, page1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, page2, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, actiontype1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, actiontype2, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, actiontype3, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, actiontype4, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, active, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, retention1, on=['register_day'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, retention2, on=['register_day'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, label, on=['user_id'], how='left')
    data.label = data.label.apply(lambda x: x if x == 1 else 0)

    return data


def get_test_data(current_day):
    # 启动特征群
    launch1 = get_launch_feature1(current_day)
    launch2 = get_launch_feature2(current_day)
    launch3 = get_launch_feature3(current_day)
    # 拍摄特征群
    video1 = get_video_feature1(current_day)
    video2 = get_video_feature2(current_day)
    video3 = get_video_feature3(current_day)
    video4 = get_video_feature4(current_day)
    # 行为特征群
    action1 = get_action_feature1(current_day)
    action2 = get_action_feature2(current_day)
    action3 = get_action_feature3(current_day)
    # 行为page特征群
    page1 = get_action_page_feature1(current_day)
    page2 = get_action_page_feature2(current_day)
    # 行为actiontype特征群
    actiontype1 = get_action_actiontype_feature1(current_day)
    actiontype2 = get_action_actiontype_feature2(current_day)
    actiontype3 = get_action_actiontype_feature3(current_day)
    actiontype4 = get_action_actiontype_feature4(current_day)
    # 注册特征群
    register = get_register_feature(current_day)
    # 活跃特征群
    active = get_active_feature(current_day)
    # 留存率
    retention1 = get_retention_feature1(current_day)
    retention2 = get_retention_feature2(current_day)

    data = pd.merge(register, launch1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, launch2, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, launch3, on=['user_id'], how='left')
    data = data.fillna(-1)

    data = pd.merge(data, video1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, video2, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, video3, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, video4, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, action1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, action2, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, action3, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, page1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, page2, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, actiontype1, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, actiontype2, on=['user_id'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, actiontype3, on=['user_id'], how='left')
    data = data.fillna(-1)
    data = pd.merge(data, actiontype4, on=['user_id'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, retention1, on=['register_day'], how='left')
    data = data.fillna(0)
    data = pd.merge(data, retention2, on=['register_day'], how='left')
    data = data.fillna(0)

    data = pd.merge(data, active, on=['user_id'], how='left')
    data = data.fillna(0)
    return data


if __name__ == '__main__':
    # 训练集
    # train dataset
    # feature:1-15 label:16-22
    # feature:9-23 label:24-30
    path = r"F:\kuaishou\data\raw_data\\"
    starttime = datetime.datetime.now()
    current_days = [16, 24]
    for i, current_day in enumerate(current_days):
        train = pd.DataFrame()
        temp = get_train_data(current_day)
        train = pd.concat([train, temp], axis=0)
        train.to_csv(path + 'data' + str(i + 1) + '.csv', index=False)
        print('The data ' + str(i + 1) + ' over.\n')
        print(train.shape)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    # 测试集
    # test dataset
    # feature:12-30 label:31-37
    starttime = datetime.datetime.now()
    current_days = [31]
    test = pd.DataFrame()
    for current_day in current_days:
        temp = get_test_data(current_day)
        test = pd.concat([test, temp], axis=0)
        test.to_csv(path + 'data' + str(3) + '.csv', index=False)
        print('The data ' + str(3) + 'over.\n')
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    print(test.shape)
