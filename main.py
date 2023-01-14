import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as scb

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol, maxVol, volBar, volPer = volRange[0], volRange[1], 400, 0

# Создаем детектор
handsDetector = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)
count = 0
prev_fist = False

while (cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == 27 or not ret:
        break
    flipped = np.fliplr(frame)

    # Переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    # Распознаем
    results = handsDetector.process(flippedRGB)

    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(flippedRGB, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

        # Рисуем круг на указательном пальце
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y * flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (0, 0, 255), -1)

        # Рисуем круг на большом пальце
        x1_tip = int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1])
        y1_tip = int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])
        cv2.circle(flippedRGB, (x1_tip, y1_tip), 10, (0, 0, 250), -1)

        # Рисуем линию между большим и указательным пальцами
        cv2.line(flippedRGB, (x_tip, y_tip), (x1_tip, y1_tip), (0, 0, 255), 3)
        # Находим длину линии
        l = math.hypot(x1_tip - x_tip, y1_tip - y_tip)

        # Уровень громкости (Правая рука)
        if "Right" in str(results.multi_handedness[0]):
            vol = np.interp(l, [50, 220], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Отображаем уровень громкости
            volBar = np.interp(l, [50, 220], [400, 150])
            volPer = np.interp(l, [50, 220], [0, 100])
            cv2.rectangle(flippedRGB, (50, 150), (85, 400), (0, 0, 255), 3)
            cv2.rectangle(flippedRGB, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(flippedRGB, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

        # Уровень яркости (Левая рука)
        else:
            br = np.interp(l, [50, 220], [0, 100])
            val = np.interp(l, [50, 220], [400, 150])

            scb.set_brightness(br)

            # Отображаем уровень яркости
            cv2.rectangle(flippedRGB, (50, 150), (85, 400), (0, 0, 255), 3)
            cv2.rectangle(flippedRGB, (50, int(val)), (85, 400), (0, 0, 255), -1)
            cv2.putText(flippedRGB, f'{int(br)} %', (20, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)


    # Переводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)
    # print(results.multi_handedness)

# Освобождаем ресурсы
handsDetector.close()