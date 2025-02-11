TypeDict = {
    '111': '황색-단선-실선',
    '112': '황색-단선-점선',
    '113': '황색-단선-좌점혼선',
    '114': '황색-단선-우점혼선',
    '121': '황색-겹선-실선',
    '122': '황색-겹선-점선',
    '123': '황색-겹선-좌점혼선',
    '124': '황색-겹선-우점혼선',
    '211': '백색-단선-실선',
    '212': '백색-단선-점선',
    '213': '백색-단선-좌점혼선',
    '214': '백색-단선-우점혼선',
    '999': '기타',
    '221': '백색-겹선-실선',
    '222': '백색-겹선-점선',
    '223': '백색-겹선-좌점혼선',
    '224': '백색-겹선-우점혼선',
    '311': '청색-단선-실선',
    '312': '청색-단선-점선',
    '313': '청색-단선-좌점혼선',
    '314': '청색-단선-우점혼선',
    '321': '청색-겹선-실선',
    '322': '청색-겹선-점선',
    '323': '청색-겹선-좌점혼선',
    '324': '청색-겹선-우점혼선',

    '1': '화살표',
    '5': '횡단보도',
}

KindDict = {
    '501': '중앙선',
    '5011': '가변차선',
    '502': '유턴구역선',
    '503': '차선',
    '504': '버스전용차선',
    '505': '길가장자리구역선',
    '599': '기타',
    '506': '진로변경제한선',
    '515': '주정차금지선',
    '525': '유도선',
    '530': '정지선', #
    '531': '안전지대', #
    '535': '자전거도로',

    '524': '정차금지대',
    '5321': '횡단보도',
    '533': '고원식횡단보도',
    '534': '자전거횡단보도',
    '5371': '직진',
    '5372': '좌회전',
    '5373': '우회전',
    '5374': '좌우회전',
    '5379': '전방향',
    '5381': '직진 및 좌회전',
    '5382': '직진 및 우회전',
    '5383': '직진 및 유턴',
    '5391': '유턴',
    '5392': '좌회전 및 유턴',
    '5431': '차로변경(좌로합류)',
    '5432': '차로변경(우로합류)',
    '544': '오르막경사면',
}

#supercategory
TypeDict_English = {
    '황색-단선-실선': 'yellow_single_solid',
    '황색-단선-점선': 'yellow_single_dashed',
    '황색-단선-좌점혼선': 'yellow_single_leftdashed_rightsolid',
    '황색-단선-우점혼선': 'yellow_single_leftsolid_rightdashed',
    '황색-겹선-실선': 'yellow_double_solid',
    '황색-겹선-점선': 'yellow_double_dashed',
    '황색-겹선-좌점혼선': 'yellow_double_leftdashed_rightsolid',
    '황색-겹선-우점혼선': 'yellow_double_leftsolid_rightdashed',
    '백색-단선-실선': 'white_single_solid',
    '백색-단선-점선': 'white_single_dashed',
    '백색-단선-좌점혼선': 'white_single_leftdashed_rightsolid',
    '백색-단선-우점혼선': 'white_single_leftsolid_rightdashed',
    '기타': 'others',
    '백색-겹선-실선': 'white_double_solid',
    '백색-겹선-점선': 'white_double_dashed',
    '백색-겹선-좌점혼선': 'white_double_leftdashed_rightsolid',
    '백색-겹선-우점혼선': 'white_double_leftsolid_rightdashed',
    '청색-단선-실선': 'blue_single_solid',
    '청색-단선-점선': 'blue_single_dashed',
    '청색-단선-좌점혼선': 'blue_single_leftdashed_rightsolid',
    '청색-단선-우점혼선': 'blue_single_leftsolid_rightdashed',
    '청색-겹선-실선': 'blue_double_solid',
    '청색-겹선-점선': 'blue_double_dashed',
    '청색-겹선-좌점혼선': 'blue_double_leftdashed_rightsolid',
    '청색-겹선-우점혼선': 'blue_double_leftsolid_rightdashed',
    '화살표': 'arrow',
    '횡단보도': 'crosswalk'
}

# category
KindDict_English = {
    '중앙선': 'center_line',
    '가변차선': 'variable_lane',
    '유턴구역선': 'u_turn_zone_line',
    '차선': 'lane_line',
    '버스전용차선': 'bus_only_lane',
    '길가장자리구역선': 'edge_line',
    '기타': 'others',
    '진로변경제한선': 'path_change_restriction_line',
    '주정차금지선': 'no_parking_stopping_line',
    '유도선': 'guiding_line',
    '정지선': 'stop_line',
    '안전지대': 'safety_zone',
    '자전거도로': 'bicycle_lane',
    '정차금지대': 'no_stopping_zone',
    '횡단보도': 'crosswalk',
    '고원식횡단보도': 'raised_crosswalk',
    '자전거횡단보도': 'bicycle_crosswalk',
    '직진': 'straight',
    '좌회전': 'left_turn',
    '우회전': 'right_turn',
    '좌우회전': 'left_right_turn',
    '전방향': 'all_directions',
    '직진 및 좌회전': 'straight_and_left_turn',
    '직진 및 우회전': 'straight_and_right_turn',
    '직진 및 유턴': 'straight_and_u_turn',
    '유턴': 'u_turn',
    '좌회전 및 유턴': 'left_turn_and_u_turn',
    '차로변경(좌로합류)': 'lane_change_merge_left',
    '차로변경(우로합류)': 'lane_change_merge_right',
    '오르막경사면': 'uphill_slope'
}