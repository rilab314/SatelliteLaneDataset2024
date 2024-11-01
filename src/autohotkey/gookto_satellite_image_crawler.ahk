F2::
CoordMode, Pixel, Screen
CoordMode, Mouse, Screen


; 텍스트 파일 경로 지정
filePath := "C:\Projects\AutoHotkey\AutoHotkey_auros노트북\400x400array_data.txt"

; 파일에서 데이터 읽기
FileRead, fileContent, %filePath%

; 데이터의 앞뒤 대괄호 제거
fileContent := Trim(fileContent, "[]")

; 콤마로 구분된 각 아이템을 배열에 저장
coordArray := StrSplit(fileContent, "], [")


repeatnum := 0
image_num := 11488
additional_num := 499
Loop
{
	image_num += 1
	not_loading := 0
	re_f5 := false

	coord := StrReplace(coordArray[image_num], ["[", "]"])
	lonlat := StrSplit(coord, ", ")
	lon := lonlat[1]
	lat := lonlat[2]

	Click, 337, 336
	Sleep, 100
	Click, 211, 406
	Sleep, 20
	Click, 211, 406
	Send, %lon%
	Sleep, 50
	Click, 361, 405
	Sleep, 20
	Click, 361, 405
	Send, %lat%
	Sleep, 50
	Click, 196, 451
	Sleep, 2000

	; +버튼 클릭하여 확대
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
		ImageSearch, X, Y, 1800, 680, 1900, 1000, *100 C:\Projects\AutoHotkey\AutoHotkey_auros노트북\images\국토위성이미지_확대완료.png
		if (ErrorLevel = 0)
		{
			break
		}
		else
		{
			Click, 1871, 744
			Sleep, 800

		}
		if (not_loading > 15)
		{
			re_f5 = true
			break
		}
	}
	if (re_f5)
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
		image_num -= 1
		continue
	}

	Click, 147, 336
	Sleep, 100

	Click, Right, 723, 499
	Sleep, 2000
	Loop
	{
		; 우클릭 메뉴뜨면 클릭
		ImageSearch, X, Y, 400, 0, 1300, 1000, *100 C:\Projects\AutoHotkey\AutoHotkey_auros노트북\images\국토위성이미지_우클릭.png
		if (ErrorLevel = 0)
		{
			Click, 721, 534
			Sleep, 2000
			break
		}
		Sleep, 300
	}

	Sleep, 2000
	formattedNum := Format("{:07}", (image_num+additional_num))
	oldPath := "C:\Users\신강민\Downloads\bies.png"
	newPath := "C:\Projects\국토지리플랫폼\국토위성이미지_크롤러\" . formattedNum . "_" . lon . "," . lat . ".png"
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
	if (repeatnum > 10)
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


F3::
ExitApp





