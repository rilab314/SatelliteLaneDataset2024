
#Persistent  ; 스크립트를 지속적으로 실행
CoordMode, Pixel, Screen
CoordMode, Mouse, Screen
SetTimer, ShowMousePos, 100  ; 100밀리초마다 ShowMousePos 레이블을 호출

ShowMousePos:
MouseGetPos, posX, posY  ; 현재 마우스 위치를 posX, posY에 저장
ToolTip, X: %posX% Y: %posY%  ; 툴팁을 통해 좌표를 표시
return




