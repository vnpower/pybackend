import asyncio
import websockets
import cv2
import numpy as np
import json
import random
import datetime

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# ARUCO_PARAMETERS = cv2.aruco.DetectorParameters_create()
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

calibration_matrix = None  
recording = False  
video_writer = None 

async def handler(websocket, path):
    global recording, video_writer

    print("connected")

    
    async def send_aruco_data():
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            aruco_result = []

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                ids = ids.flatten()
                for id, corner in zip(ids, corners):
        
                    corners = corner.reshape((4, 2))
                    cx = int((corners[0][0] + corners[2][0]) / 2.0)
                    cy = int((corners[0][1] + corners[2][1]) / 2.0)

                    if calibration_matrix is not None:
                        point = np.array([[cx, cy]], dtype='float32')
                        remapped_point = cv2.perspectiveTransform(np.array([point]), calibration_matrix)
                        cx, cy = remapped_point[0][0]
                        cx, cy = int(cx), int(cy)
                        if id == 4 : print(f"Remapped point: {cx}, {cy}")

                    aruco_result.append({"id": int(id), "cx": cx, "cy": cy})

            if aruco_result:
                aruco_data = json.dumps({"type": "aruco", "data": aruco_result})
                await websocket.send(aruco_data)

            if recording and video_writer:
                video_writer.write(frame)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(10)
            if key == 27:
                break

            await asyncio.sleep(0.1)

    async def receive_messages():
        global recording, video_writer
        while True:
            try:
                message = await websocket.recv()
                print(f"Received message from client: {message.strip()}")

                if message.lower() == "calibrate":
                    await calibrate()
                elif message.lower() == "start video":
                    await start_recording()
                elif message.lower() == "save video":
                    await stop_recording()
            except websockets.exceptions.ConnectionClosedError:
                print("Connection closed")
                break
            except Exception as e:
                print(f"An error occurred: {e}")


    async def calibrate():
        global calibration_matrix
        print("Starting calibration...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                ids = ids.flatten()
                corner_points = {}


                expected_ids = {0: "Top-left", 1: "Top-right", 2: "Bottom-right", 3: "Bottom-left"}
                
                for id, corner in zip(ids, corners):
                    if id in expected_ids:
                        corners = corner.reshape((4, 2))
                        # x_br, y_br = corners[2]
                        # x_tl, y_tl = corners[0]  # Top-left of the marker

                        # cx = (x_br-x_tl)/2 + x_tl
                        # cy = (y_br-y_tl)/2 +y_tl
                        # print(id, cx, cy)
                        if id == 0:  # Top-left of calibration area
                            cx, cy = corners[0]  # Bottom-left of the marker
                        elif id == 1:  # Top-right of calibration area
                            cx, cy = corners[1]  # Bottom-right of the marker
                        elif id == 2:  # Bottom-right of calibration area
                            cx, cy = corners[2]  # Top-right of the marker
                        elif id == 3:  # Bottom-left of calibration area
                            cx, cy = corners[3]  # Top-left of the marker

                        corner_points[id] = (cx, cy)

                if len(corner_points) == 4:

                    src_points = np.float32([
                        corner_points[0],  # Top-left
                        corner_points[1],  # Top-right
                        corner_points[2],  # Bottom-right
                        corner_points[3]   # Bottom-left
                    ])
                    dst_points = np.float32([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
                    
                    calibration_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    print("Calibration completed!")
                    break
                else :
                    print("Not all corners detected")


            await asyncio.sleep(0.1)

    async def start_recording():
        global recording, video_writer
        if not recording:
            print("Starting video recording...")
            
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi" 
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 720))  # Set FPS to 60.0
            recording = True

    async def stop_recording():
        global recording, video_writer
        if recording:
            print("Stopping video recording...")
            recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None

    send_task = asyncio.create_task(send_aruco_data())
    receive_task = asyncio.create_task(receive_messages())
    await asyncio.gather(send_task, receive_task)

async def main():
    async with websockets.serve(handler, "localhost", 3030):
        await asyncio.Future()

asyncio.run(main())
