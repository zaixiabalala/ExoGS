import subprocess
import argparse
from pathlib import Path


def video_to_images(video_path, output_dir, fps=None, image_format='jpg', quality=2):
    """
    将视频转换为图像序列
    
    参数:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出图像目录
        fps (float, optional): 提取帧率，如果为None则提取所有帧
        image_format (str): 输出图像格式，默认为'jpg'
        quality (int): JPEG质量 (2-31，2为最高质量)，PNG时忽略
    
    返回:
        bool: 转换是否成功
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # 检查视频文件是否存在
    if not video_path.exists():
        print(f"[ERROR] 视频文件不存在: {video_path}")
        return False
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建ffmpeg命令
    output_pattern = output_dir / f"frame_%06d.{image_format}"
    
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-y',  # 覆盖输出文件
    ]
    
    # 如果指定了fps，添加帧率参数
    if fps is not None:
        cmd.extend(['-r', str(fps)])
    
    # 添加输出参数
    if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
        cmd.extend([
            '-q:v', str(quality),  # JPEG质量
            str(output_pattern)
        ])
    else:
        cmd.append(str(output_pattern))
    
    try:
        # 执行ffmpeg命令
        print(f"[INFO] 正在将视频转换为图像...")
        print(f"[INFO] 输入视频: {video_path}")
        print(f"[INFO] 输出目录: {output_dir}")
        if fps:
            print(f"[INFO] 提取帧率: {fps} fps")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # 统计生成的图像数量
        image_count = len(list(output_dir.glob(f"*.{image_format}")))
        print(f"[INFO] 成功生成 {image_count} 张图像")
        return True
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[ERROR] ffmpeg执行失败: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("[ERROR] 未找到ffmpeg，请确保已安装ffmpeg")


def main():
    # ========== 在这里手动设置参数 ==========
    # 环境：base
    video_path = '/media/ubuntu/B0A8C06FA8C0361E/Data/Colmap_Data/paper_box/paper_box.MOV'  # 输入视频文件路径
    output_dir = '/media/ubuntu/B0A8C06FA8C0361E/Data/Colmap_Data/paper_box/images'  # 输出目录，None则使用默认名称（视频文件名_images）
    fps = 1.5  # 提取帧率，None则提取所有帧，例如: 1.0 表示每秒1帧
    image_format = 'jpg'  # 输出图像格式: 'jpg', 'jpeg', 或 'png'
    quality = 2  # JPEG质量 (2-31，2为最高质量)
    # ========================================
    # 执行转换
    success = video_to_images(
        video_path,
        output_dir,
        fps=fps,
        image_format=image_format,
        quality=quality
    )
    if success:
        print("[INFO] 转换完成！")
    else:
        print("[ERROR] 转换失败！")
        exit(1)


if __name__ == '__main__':
    main()