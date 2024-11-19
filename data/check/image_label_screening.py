import os
import cv2
import glob
import shutil

# 이미지가 저장된 폴더 경로
folder_path = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/drawn_image"  # "a" 폴더가 현재 디렉토리에 있어야 합니다.
destination_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/false_image"  # 이동할 대상 폴더
break_image_folder = "/media/falcon/50fe2d19-4535-4db4-85fb-6970f063a4a11/Ongoing/2024_SATELLITE/datasets/total_origin_lonlat/break_image"  # 이동할 대상 폴더
# b 폴더가 없으면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# .png 파일만 불러오기
images = sorted(glob.glob(os.path.join(folder_path, "*.png")))
current_index = 7621

if not images:
    print("폴더에 .png 이미지가 없습니다.")
    exit()

while True:
    # 현재 이미지 경로
    current_image_path = images[current_index]

    # 이미지 읽기
    image = cv2.imread(current_image_path)
    if image is None:
        print(f"이미지 {current_image_path}를 열 수 없습니다.")
        break

    scale = 1.2
    resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 확대된 이미지 표시
    print(f"현재 이미지: {os.path.basename(images[current_index])} (총 {len(images)}개 중 {current_index + 1}번째)")
    cv2.imshow("Image Viewer", resized_image)

    # 키 입력 대기
    key = cv2.waitKey(0)

    if key == ord('q'):  # q를 누르면 b 폴더로 이동
        destination_path = os.path.join(destination_folder, os.path.basename(current_image_path))
        print(f"이미지 {os.path.basename(images[current_index])}를 {destination_folder} 폴더로 이동")
        shutil.move(current_image_path, destination_path)
        del images[current_index]

        if not images:  # 모든 이미지가 이동된 경우 종료
            print("모든 이미지가 이동되었습니다.")
            break

    elif key == ord('w'):  # q를 누르면 c 폴더로 이동
        break_image_path = os.path.join(break_image_folder, os.path.basename(current_image_path))
        print(f"이미지 {os.path.basename(images[current_index])}를 {break_image_folder} 폴더로 이동")
        shutil.move(current_image_path, break_image_path)
        del images[current_index]

        if not images:  # 모든 이미지가 이동된 경우 종료
            print("모든 이미지가 이동되었습니다.")
            break
    elif key == ord('d'):  # d를 누르면 다음 이미지로 이동
        current_index = (current_index + 1) % len(images)
    elif key == ord('a'):  # a를 누르면 이전 이미지로 이동
        current_index = (current_index - 1) % len(images)
    elif key == 27:  # ESC를 누르면 종료
        print("프로그램 종료")
        break

cv2.destroyAllWindows()
