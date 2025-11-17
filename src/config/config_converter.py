ADE20K_LANE_CATEGORIES = {
    '000': {'id':0, 'priority':11, 'category':'ignore',},
    '501': {'id':1, 'priority':10, 'category':'center_line',},
    '502': {'id':2, 'priority':6, 'category':'u_turn_zone_line',},
    '503': {'id':3, 'priority':7, 'category':'lane_line',},
    '504': {'id':4, 'priority':3, 'category':'bus_only_lane',},
    '505': {'id':5, 'priority':8, 'category':'edge_line',},
    '506': {'id':6, 'priority':4, 'category':'path_change_restriction_line',},
    '515': {'id':7, 'priority':5, 'category':'no_parking_stopping_line',},
    '525': {'id':8, 'priority':9, 'category':'guiding_line',},
    '530': {'id':9, 'priority':0, 'category':'stop_line',},
    '531': {'id':10, 'priority':1, 'category':'safety_zone',},
    '535': {'id':11, 'priority':2, 'category':'bicycle_lane'},
}


COCO_OD_CATEGORIES = {
    '5321': 'crosswalk',
    '533': 'raised_crosswalk',
    '534': 'bicycle_crosswalk',
    '5371': 'straight',
    '5372': 'left_turn',
    '5373': 'right_turn',
    '5381': 'straight_and_left_turn',
    '5382': 'straight_and_right_turn',
    '5383': 'straight_and_u_turn',
    '5391': 'u_turn',
    '5392': 'left_turn_and_u_turn',
    '5431': 'lane_change_merge_left',
    '5432': 'lane_change_merge_right',
    '544': 'uphill_slope',
}