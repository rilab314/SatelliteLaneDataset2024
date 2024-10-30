import os
import json

if __name__ == '__main__':
    # coco_data = "/media/falcon/IanBook8T1/ArchiveDrive/PublicDataset/unzips/coco/annotations/instances_train2017.json"
    # with open(coco_data, 'r', encoding='utf-8') as file:
    #     data = json.load(file)

    satellite_data_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/archive/국토정보플랫폼/total_road_links.json"
    with open(satellite_data_path, 'r', encoding='utf-8') as file:
        satellite_data = json.load(file)


    for i in satellite_data["categories"]:
        print(f"\'{i['name']}\', ", end="")
# ('center_line', 'variable_lane', 'u_turn_zone_line', 'lane_line', 'bus_only_lane', 'edge_line', 'others',
#  'path_change_restriction_line', 'no_parking_stopping_line', 'guiding_line', 'stop_line', 'safety_zone',
#  'bicycle_lane', 'no_stopping_zone', 'crosswalk', 'raised_crosswalk', 'bicycle_crosswalk', 'straight', 'left_turn',
#  'right_turn', 'left_right_turn', 'all_directions', 'straight_and_left_turn', 'straight_and_right_turn',
#  'straight_and_u_turn', 'u_turn', 'left_turn_and_u_turn', 'lane_change_merge_left', 'lane_change_merge_right',
#  'uphill_slope')