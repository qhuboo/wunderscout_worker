import threading
import time
import torch


def process_video(local_path, output_path, job_id, pipeline):
    print(
        f"[process_video][{time.strftime('%X')}] Job {job_id} running in thread {threading.get_ident()}"
    )
    if torch.cuda.is_available():
        print("GPU name: ", torch.cuda.get_device_name(0))
        # pipeline.run(local_path, output_path)
    else:
        print("No GPU Access")
