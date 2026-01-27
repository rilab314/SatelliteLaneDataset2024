F2::	; F2 버튼 누르면 아래 코드 시작
CoordMode, Pixel, Screen
CoordMode, Mouse, Screen


; 텍스트 파일 경로 지정
filePath := "C:\Users\rkdal\Downloads\SatelliteLaneDataset2024-main\SatelliteLaneDataset2024-main\src\autohotkey\400x400array_data.txt"

; 파일에서 데이터 읽기
FileRead, fileContent, %filePath%

; 데이터의 앞뒤 대괄호 제거
fileContent := Trim(fileContent, "[]")

; 콤마로 구분된 각 아이템을 배열에 저장
coordArray := StrSplit(fileContent, "], [")


repeatnum := 0	; 반복 횟수(10번 이상 반복 시 F5)
image_num := 0	; 몇번까지 이미지를 받았는지
additional_num := 0
Loop
{
	image_num += 1
	not_loading := 0
	re_f5 := false

	coord := StrReplace(coordArray[image_num], ["[", "]"])
	lonlat := StrSplit(coord, ", ")
	lon := lonlat[1]
	lat := lonlat[2]

	; 반경
	Click, 337, 336
	Sleep, 100
	; 경도
	Click, 211, 406
	Sleep, 20
	Click, 211, 406
	Send, %lon%
	Sleep, 50
	; 위도
	Click, 361, 405
	Sleep, 20
	Click, 361, 405
	Send, %lat%
	Sleep, 50
	; "1km"버튼
	Click, 196, 451
	Sleep, 2000

	; 우하단 +버튼 클릭하여 확대
	Click, 1871, 744
	Sleep, 800
	Click, 1871, 744
	Sleep, 800
	Click, 1871, 744
	Sleep, 800
	Click, 1871, 744
	Sleep, 800

	Loop
	{
		not_loading += 1
		; 우클릭 메뉴뜨면 클릭
		ImageSearch, X, Y, 1800, 680, 1900, 1000, *100 C:\Users\rkdal\Downloads\SatelliteLaneDataset2024-main\SatelliteLaneDataset2024-main\src\autohotkey\images\국토위성이미지_확대완료.png
		if (ErrorLevel = 0)
		{
			break
		}
		else
		{
			Click, 1871, 744	; 확대가 다 안 됐을 때 다시 확대
			Sleep, 800

		}
		if (not_loading > 15)	; 15초 정도 확대가 안되면 F5
		{
			re_f5 = true
			break
		}
	}
	if (re_f5)	; 리셋
	{
		Send, {F5}
		Sleep, 3500
		Click, 900, 600		; 땅바닥 클릭
		Sleep, 200
		Click, 1809, 214	; 바탕화면 선택 클릭
		Sleep, 100
		Click ,1741, 245	; 영상지도 클릭
		Sleep, 100
		Click, 1738, 278	;하이브리드 체크 해제
		image_num -= 1
		continue
	}


	Click, 147, 336
	Sleep, 100
	; 화면 중앙 우클릭
	Click, Right, 723, 499
	Sleep, 2000
	Loop
	{
		; 우클릭 메뉴뜨면 클릭
		ImageSearch, X, Y, 400, 0, 1300, 1000, *100 C:\Users\rkdal\Downloads\SatelliteLaneDataset2024-main\SatelliteLaneDataset2024-main\src\autohotkey\images\국토위성이미지_우클릭.png
		if (ErrorLevel = 0)
		{
			Click, 721, 534	; 이미지 저장 버튼
			Sleep, 2000
			break
		}
		Sleep, 300
	}

	Sleep, 2000
	formattedNum := Format("{:07}", (image_num+additional_num))
	; 저장된 이미지의 경로와 새로 저장할 이미지 경로 지정하여 새 경로에 이미지 저장
	oldPath := "C:\Users\rkdal\Downloads\bies.png"
	newPath := "C:\Users\rkdal\Desktop\국토위성이미지_크롤러\" . formattedNum . "_" . lon . "," . lat . ".png"
	FileMove, %oldPath%, %newPath%
	; 파일이 존재하는지 확인
	Loop
	{
		if (FileExist(oldPath))
		{
			FileMove, %oldPath%, %newPath%
		}
		else  ; 파일이 존재하지 않을 경우
		{
			break
		}
	}

	repeatnum += 1
	if (repeatnum > 10)	; 너무 많이 반복했다면 F5로 새로고침
	{
		Send, {F5}
		Sleep, 3500
		Click, 900, 600
		Sleep, 200
		Click, 1809, 214
		Sleep, 100
		Click ,1741, 245
		Sleep, 100
		Click, 1738, 278
		repeatnum := 0
	}
}


F3::	; F3 누르면 종료
ExitApp





