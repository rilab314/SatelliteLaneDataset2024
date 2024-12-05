
SHAPE_PATH = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/정밀도로지도/unzip'
DATASET_PATH = '/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/satellite_good_matching_241125'
ORIGINAL_IMAGE_PATH = DATASET_PATH + '/origin_image'

COORD_LIST_PATH = DATASET_PATH+'/coord_list.json'
JSON_PATH = DATASET_PATH+'/json'
UNMATCHED_LABEL_PATH = DATASET_PATH+'/unmatched_label'
IMAGE_PATH = DATASET_PATH+'/image'
LABEL_PATH = DATASET_PATH+'/label'

CUSTOM_ADE20K_PATH = DATASET_PATH+'/satellite_ade20k_241125'

SURFACE_SHAPE_endswith_NAME = '/HDMap_UTM52N_타원체고/B3_SURFACEMARK.shp'
LANE_SHAPE_endswith_NAME = '/HDMap_UTM52N_타원체고/B2_SURFACELINEMARK.shp'

SEOUL_CONFIG = {
    'LONGITUDE_RANGE': [126.8255, 127.1516],
    'LATITUDE_RANGE': [37.4738, 37.6202],
    'LONGITUDE_STRIDE': 0.00092,
    'LATITUDE_STRIDE': 0.000764,
    'REGION': 'Seoul, Korea',
}
INCHEON_CONFIG = {
    'LONGITUDE_RANGE': [126.5987, 126.8068],
    'LATITUDE_RANGE': [37.3648, 37.5910],
    'LONGITUDE_STRIDE': 0.001107072,
    'LATITUDE_STRIDE': 0.00088247424,
    'REGION': 'Incheon, Korea',
}

MAIN_REGION_CFG = SEOUL_CONFIG

LONGITUDE_RANGE = MAIN_REGION_CFG['LONGITUDE_RANGE']
LATITUDE_RANGE = MAIN_REGION_CFG['LATITUDE_RANGE']
LONGITUDE_STRIDE = MAIN_REGION_CFG['LONGITUDE_STRIDE']
LATITUDE_STRIDE = MAIN_REGION_CFG['LATITUDE_STRIDE']
REGION = MAIN_REGION_CFG['REGION']

ONE_PIXEL_LONGITUDE = 0.002641 / 1832
ONE_PIXEL_LATITUDE = 0.0010985 / 956
IMAGE_LONGITUDE_DIM = 0
IMAGE_LATITUDE_DIM = 0
# LONGITUDE_STRIDE = IMAGE_LONGITUDE_DIM / 2.
# LATITUDE_STRIDE = IMAGE_LATITUDE_DIM / 2.
IMAGE_SIZE_w = 768
IMAGE_SIZE_h = 768

DATASET_RATIO = {'train': 0.7,
                 'validation': 0.1,
                 'test': 0.2}


TypeDict = {
    '111': 'yellow_single_solid',
    '112': 'yellow_single_dashed',
    '113': 'yellow_single_leftdashed_rightsolid',
    '114': 'yellow_single_leftsolid_rightdashed',
    '121': 'yellow_double_solid',
    '122': 'yellow_double_dashed',
    '123': 'yellow_double_leftdashed_rightsolid',
    '124': 'yellow_double_leftsolid_rightdashed',
    '211': 'white_single_solid',
    '212': 'white_single_dashed',
    '213': 'white_single_leftdashed_rightsolid',
    '214': 'white_single_leftsolid_rightdashed',
    '999': 'others',
    '221': 'white_double_solid',
    '222': 'white_double_dashed',
    '223': 'white_double_leftdashed_rightsolid',
    '224': 'white_double_leftsolid_rightdashed',
    '311': 'blue_single_solid',
    '312': 'blue_single_dashed',
    '313': 'blue_single_leftdashed_rightsolid',
    '314': 'blue_single_leftsolid_rightdashed',
    '321': 'blue_double_solid',
    '322': 'blue_double_dashed',
    '323': 'blue_double_leftdashed_rightsolid',
    '324': 'blue_double_leftsolid_rightdashed',

    '1': 'arrow',
    '5': 'crosswalk',
    None: 'None',
    '101': 'undefined'
}

KindDict = {
    '501': 'center_line',
    '5011': 'variable_lane',
    '502': 'u_turn_zone_line',
    '503': 'lane_line',
    '504': 'bus_only_lane',
    '505': 'edge_line',
    '599': 'others',
    '506': 'path_change_restriction_line',
    '515': 'no_parking_stopping_line',
    '525': 'guiding_line',
    '530': 'stop_line',
    '531': 'safety_zone',
    '535': 'bicycle_lane',

    '524': 'no_stopping_zone',
    '5321': 'crosswalk',
    '533': 'raised_crosswalk',
    '534': 'bicycle_crosswalk',
    '5371': 'straight',
    '5372': 'left_turn',
    '5373': 'right_turn',
    '5374': 'left_right_turn',
    '5379': 'all_directions',
    '5381': 'straight_and_left_turn',
    '5382': 'straight_and_right_turn',
    '5383': 'straight_and_u_turn',
    '5391': 'u_turn',
    '5392': 'left_turn_and_u_turn',
    '5431': 'lane_change_merge_left',
    '5432': 'lane_change_merge_right',
    '544': 'uphill_slope',
    None: 'None'
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
