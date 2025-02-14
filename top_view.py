import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


class GoProTopViewProcessor:
    def __init__(self, video_path, use_gpu=True):
        """Initialize video processor"""
        import os
        os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '50000'

        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            print("Using GPU acceleration")
        else:
            print("Using CPU only")

        self.video = cv2.VideoCapture(video_path)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # バッファサイズを増やす

        if not self.video.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties with error checking
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # GoPro camera parameters (4K)
        fov = 90.0  # GoPro's typical FOV in degrees
        fx = self.width / (2 * np.tan(np.radians(fov / 2)))
        fy = fx  # 正方形ピクセルを仮定
        cx = self.width / 2
        cy = self.height / 2

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        # カメラの設置パラメータ
        self.camera_height = 1.15  # メートル
        self.camera_pitch = np.radians(0)  # 水平方向
        self.camera_yaw = np.radians(0)  # 前方向

        # トップビュー変換のパラメータ
        self.road_width = 3.5  # メートル（一般的な車線幅）
        self.view_distance = 20.0  # メートル（見たい前方距離）

    def calculate_projection_matrix(self):
        """Calculate projection matrix for top view transformation"""
        # 回転行列
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.camera_pitch), -np.sin(self.camera_pitch)],
            [0, np.sin(self.camera_pitch), np.cos(self.camera_pitch)]
        ])

        # 平行移動ベクトル
        t = np.array([[0], [self.camera_height], [0]])

        # 外部パラメータ行列 [R|t]
        RT = np.hstack((Rx, t))

        # 射影行列
        P = self.camera_matrix @ RT

        return P

    def get_transformation_points(self, output_size):
        """Calculate source and destination points for perspective transform"""
        # 入力画像上の点（台形の4点）
        vanishing_point_y = int(self.height * 0.7)  # 消失点のy座標
        top_width = self.width * 0.4  # 台形上部の幅

        src_pts = np.float32([
            [self.width / 2 - top_width / 2, vanishing_point_y],  # 上左
            [self.width / 2 + top_width / 2, vanishing_point_y],  # 上右
            [0, self.height],  # 下左
            [self.width, self.height]  # 下右
        ])

        # 出力画像上の点（長方形の4点）
        margin = 100  # ピクセルマージン
        dst_pts = np.float32([
            [margin, margin],
            [output_size[0] - margin, margin],
            [margin, output_size[1] - margin],
            [output_size[0] - margin, output_size[1] - margin]
        ])

        return src_pts, dst_pts

    def transform_frame(self, frame, output_size=(1920, 1080)):
        """Transform frame to top view"""
        src_pts, dst_pts = self.get_transformation_points(output_size)

        if self.use_gpu:
            # GPU版の処理
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # 変換行列を計算
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            gpu_M = cv2.cuda_GpuMat()
            gpu_M.upload(M.astype(np.float32))

            # 射影変換を適用
            gpu_warped = cv2.cuda.warpPerspective(gpu_frame, gpu_M, output_size)
            result = gpu_warped.download()

            return result
        else:
            # CPU版の処理
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, M, output_size)

            return warped

    def process_video(self, output_dir, interval_sec=1.0):
        """Process video and save frames"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frame_interval = int(self.fps * interval_sec)
        total_frames = self.frame_count // frame_interval

        print(f"Processing video: {total_frames} frames")

        frame_num = 0
        processed_count = 0

        while processed_count < total_frames:
            # フレーム位置を設定
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # フレームの読み込みを試行
            ret = False
            for _ in range(3):  # 最大3回試行
                ret, frame = self.video.read()
                if ret:
                    break

            if ret:
                # Transform frame
                top_view = self.transform_frame(frame)

                # Save frame
                frame_path = output_path / f"frame_{processed_count:06d}.jpg"
                cv2.imwrite(str(frame_path), top_view)
                processed_count += 1

            frame_num += frame_interval
            if frame_num >= self.frame_count:
                break

        self.video.release()
        print("Processing complete!")


def main():
    # GoProのマルチストリーム対応
    import os
    os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

    parser = argparse.ArgumentParser(description="Transform GoPro video to top view frames")
    parser.add_argument("video_path", help="Path to GoPro video file")
    parser.add_argument("output_dir", help="Directory to save output frames")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Interval between frames (seconds)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")

    args = parser.parse_args()

    processor = GoProTopViewProcessor(args.video_path, use_gpu=not args.no_gpu)
    processor.process_video(args.output_dir, args.interval)


if __name__ == "__main__":
    main()