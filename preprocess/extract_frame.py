import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

input_dir = "/path/to/Charades/VideoData"   # 输入视频文件夹
output_dir = "/path/to/Charades/ImageData"  # 输出帧文件夹
fps = 2                  # 每秒抽取多少帧
num_workers = 20          # 并行进程数，可根据机器 GPU/CPU 调整

os.makedirs(output_dir, exist_ok=True)


def process_video(file):
    if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv")):
        return None

    video_path = os.path.join(input_dir, file)
    video_id = os.path.splitext(file)[0]
    save_dir = os.path.join(output_dir, video_id)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"无法打开视频: {video_path}"

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(round(video_fps / fps)) if video_fps > 0 else 1

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            save_path = os.path.join(save_dir, f"{video_id}_{saved_count:06d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return f"{video_id}: 抽取 {saved_count} 帧，保存到 {save_dir}"


if __name__ == "__main__":
    files = os.listdir(input_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, f): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                print(result)
