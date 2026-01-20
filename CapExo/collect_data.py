import os
from datetime import datetime
from typing import Tuple
from tap import Tap

from r3kit.devices.encoder.pdcd.angler import Angler
from r3kit.devices.camera.realsense.d415 import D415


class ArgumentParser(Tap):
    encoder_id: str = '/dev/ttyUSB0'
    encoder_index: Tuple[str, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    encoder_baudrate: int = 1000000
    encoder_name: str = 'Angler'

    camera_id: str = None
    camera_depth: bool = False
    camera_name: str = 'D415'

    save_path: str = "/media/ubuntu/B0A8C06FA8C0361E/Data/Origin_Data/records_1208_test"


def main(args: ArgumentParser):
    print("Initializing...")
    encoder = Angler(id=args.encoder_id, index=args.encoder_index, fps=0, baudrate=args.encoder_baudrate,
                     gap=-1, strict=True, name=args.encoder_name)
    camera = D415(id=args.camera_id, depth=args.camera_depth, name=args.camera_name)

    input("Press Enter to start collection...")
    os.makedirs(args.save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    encoder_save_path = os.path.join(save_path, 'encoder')
    camera_save_path = os.path.join(save_path, 'camera')
    os.makedirs(encoder_save_path, exist_ok=True)
    os.makedirs(camera_save_path, exist_ok=True)
    camera.set_streaming_save_path(camera_save_path)

    encoder.start_streaming()
    camera.start_streaming()
    encoder.collect_streaming(collect=True)
    camera.collect_streaming(collect=True)

    try:
        while True:
            input("Collection started... (Press Enter to stop and save)")

            encoder.collect_streaming(collect=False)
            camera.collect_streaming(collect=False)
            print("Collection stopped, saving data...")

            encoder_data = encoder.get_streaming()
            camera_data = camera.get_streaming()

            encoder.save_streaming(save_path=encoder_save_path, streaming_data=encoder_data)
            camera.save_streaming(save_path=camera_save_path, streaming_data=camera_data)

            print(f"Encoder data saved to {encoder_save_path}, camera data saved to {camera_save_path}")
            print(f"Encoder samples: {len(encoder_data['timestamp_ms'])}, Camera samples: {len(camera_data['timestamp_ms'])}")

            continue_collect = input("Save completed. Continue collection? (y/n, default: y): ").strip().lower()
            if continue_collect == 'n':
                break

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(args.save_path, timestamp)
            os.makedirs(save_path, exist_ok=True)

            encoder_save_path = os.path.join(save_path, 'encoder')
            camera_save_path = os.path.join(save_path, 'camera')
            os.makedirs(encoder_save_path, exist_ok=True)
            os.makedirs(camera_save_path, exist_ok=True)
            camera.set_streaming_save_path(camera_save_path)

            encoder.reset_streaming()
            camera.reset_streaming()
            encoder.collect_streaming(collect=True)
            camera.collect_streaming(collect=True)

    finally:
        encoder.stop_streaming()
        camera.stop_streaming()
        print("Data collection stopped")


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    main(args)
