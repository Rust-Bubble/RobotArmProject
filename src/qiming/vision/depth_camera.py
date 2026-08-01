from __future__ import annotations

import os
import glob

from src.qiming.models import ImageFrame


class DepthCamera:
    def __init__(self, camera_cfg: dict):
        self.intrinsic_matrix = None
        self.dist_coeffs = None
        self.T_cam2end = None
        self.HandEye_calibration_img = camera_cfg.get("calibration_img_dir", r"D:\camera_images")
        self.SDK_Path = camera_cfg.get("sdk_path", r"D:\Documnet\Other material\3D摄像头客户资料\驱动程序及SDK开发\windows\OpenNI2 SDK\Windows_V2.3.0\windows\SDK\x64\Redist")
        self.all_rvec = []
        self.all_tvec = []
        self.calibration_objpoints = []
        self.calibration_imgpoints = []
        self.dev = None
        self.depth_stream = None
        self.color_stream = None
        self.fail_data = []

    def open_camera(self):
        from openni import openni2

        if not os.path.exists(self.SDK_Path):
            print(f"错误：SDK路径不存在! 当前填写的路径是: {self.SDK_Path}")
            return False
        try:
            openni2.initialize(self.SDK_Path)
            self.dev = openni2.Device.open_any()

            if self.dev.has_sensor(openni2.SENSOR_DEPTH):
                self.depth_stream = self.dev.create_depth_stream()
                self.depth_stream.start()
                print("深度流启动成功")
            else:
                print("设备无深度传感器，无法启动深度流")
                return False

            if self.dev.has_sensor(openni2.SENSOR_COLOR):
                self.color_stream = self.dev.create_color_stream()
                self.color_stream.start()
                try:
                    self.color_stream.set_auto_white_balance(True)
                    self.color_stream.set_auto_exposure(True)
                    print("已开启自动白平衡和自动曝光")
                except Exception as e:
                    print(f"设备不支持自动白平衡/曝光: {str(e)}")
                print("彩色流启动成功")
            else:
                print("设备无彩色传感器，无法启动彩色流")
                return False

            return True
        except Exception as e:
            print(f"相机初始化失败: {str(e)}")
            self.close_camera()
            return False

    def get_frames(self):
        import numpy as np
        import cv2

        if not self.dev or not self.depth_stream or not self.color_stream:
            print("错误：相机未打开或流未启动，请先调用open_camera()")
            return None, None

        try:
            depth_frame = self.depth_stream.read_frame()
            depth_data = np.array(
                depth_frame.get_buffer_as_uint16(),
                dtype=np.uint16
            ).reshape(depth_frame.height, depth_frame.width)

            color_frame = self.color_stream.read_frame()
            color_rgb = np.array(
                color_frame.get_buffer_as_uint8(),
                dtype=np.uint8
            ).reshape(color_frame.height, color_frame.width, 3)
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

            return color_bgr, depth_data
        except Exception as e:
            print(f"帧读取失败: {str(e)}")
            return None, None

    def capture(self) -> ImageFrame:
        color_frame, depth_frame = self.get_frames()
        return ImageFrame(
            data=color_frame,
            color_frame=color_frame,
            depth_frame=depth_frame
        )

    def save_current_frames(self, current_frame=None):
        import cv2

        save_path = self.HandEye_calibration_img

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
                print(f"自动创建保存目录: {save_path}")
        except Exception as e:
            print(f"创建保存目录失败: {str(e)}")
            return False

        color_img, depth_img = self.get_frames()
        if color_img is None:
            print("无有效帧数据，无法保存")
            return False

        color_save_path = os.path.join(save_path, f"{current_frame}_HandEye.png")
        try:
            cv2.imwrite(color_save_path, color_img)
        except Exception as e:
            print(f"彩色图像保存失败: {str(e)}")
            return False
        print("\n=== 图像保存成功 ===")
        print(f"彩色图: {color_save_path}")
        print("===================\n")
        return True

    def HandEye_calibration_capture(self, calib_idx):
        if not self.open_camera():
            print("相机启动失败，无法执行手眼标定")
            return

        print(f"\n手眼标定开始！")
        print("每次拍摄前请调整标定板位置（确保棋盘格完整可见）\n")

        if_success = True

        try:
            current_time = calib_idx
            print(f"【第 {current_time}次拍摄】")
            input("请调整标定板后按 Enter 键拍摄...")

            save_success = self.save_current_frames(current_time)
            if not save_success:
                print(f"第 {current_time} 次拍摄保存失败，将重新尝试")
                input("请确认标定板位置后按 Enter 键重新拍摄...")
                self.save_current_frames()

            print("\n手眼标定完成！图像已保存到:")
            print(f"{self.HandEye_calibration_img}")
        except KeyboardInterrupt:
            print("\n用户中断标定流程")
        finally:
            self.close_camera()

    def close_camera(self):
        from openni import openni2

        print("\n正在关闭相机资源...")
        try:
            if self.depth_stream and self.depth_stream.is_valid():
                self.depth_stream.stop()
                self.depth_stream.destroy()
                print("深度流已关闭")
            if self.color_stream and self.color_stream.is_valid():
                self.color_stream.stop()
                self.color_stream.destroy()
                print("彩色流已关闭")
            if self.dev and self.dev.is_valid():
                self.dev.close()
                print("相机设备已关闭")
            openni2.unload()
            print("OpenNI SDK已卸载")
        except Exception as e:
            print(f"关闭相机资源时出错: {str(e)}")

    def going_camera_calibration(self, chessboard_size=(9, 6), square_size=0.025):
        import numpy as np
        import cv2

        image_dir = self.HandEye_calibration_img
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        success_calibration_count = []

        objpoints = []
        imgpoints = []

        print(f"正在扫描图像目录: {image_dir}")
        images = glob.glob(f"{image_dir}/*.jpg")

        if not images:
            raise ValueError("未找到标定图像，请检查路径是否正确")

        success_count = 0

        for img_idx, img_path in enumerate(images, start=1):
            print(f"\n处理第{img_idx}张图片: {img_path}")

            img = cv2.imread(img_path)
            if img is None:
                error_msg = f"第{img_idx}张图片读取失败"
                print(f" {error_msg}")
                continue

            if img.size == 0:
                error_msg = f"第{img_idx}张图片为空或损坏"
                print(f" {error_msg}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners2)
                success_calibration_count.append(img_idx)
                print(f"第{img_idx}张图片处理成功")

                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.imshow("Calibration", img)
                cv2.waitKey(300)
            else:
                self.fail_data.append(img_idx)
                print(f"第{img_idx}张图片角点检测失败")

        cv2.destroyAllWindows()

        if not objpoints or not imgpoints:
            raise RuntimeError("没有有效的角点数据，无法进行标定")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        if not ret:
            raise RuntimeError("标定失败，请检查图像质量或数量")

        self.intrinsic_matrix = mtx
        self.dist_coeffs = dist

        return objpoints, imgpoints, success_calibration_count

    def get_depth_value(self, y, x):
        color_cframe, depth_cframe = self.get_frames()

        if y > 480 or x > 640 or x < 0 or y < 0:
            print("超出深度图像范围")
        else:
            return depth_cframe[y, x]

    def get_obj_in_camera(self, y, x):
        import numpy as np

        inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        obj_img_poisition = np.array([[y],
                                      [x],
                                      [1]])

        depth_value = self.get_depth_value(y, x)

        if depth_value == 0:
            print("检测目标的深度值为0！请移动物体！")

        while depth_value == 0:
            depth_value = self.get_depth_value(y, x)

        return np.dot(inverse_intrinsic_matrix, obj_img_poisition) * depth_value