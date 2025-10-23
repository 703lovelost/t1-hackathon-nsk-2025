import cv2
from queue import Empty, Full
import multiprocessing as mp
from multiprocessing import Process, Queue, Event

import numpy as np
from torchvision.transforms import ToPILImage

from segment_models import SegmentModel, YoloModel


class FrameProcessor(Process):
    def __init__(
            self,
            input_queue: Queue,
            result_queue: Queue,
            stop_event,
            model_weight: str,
            worker_id: str,
            fps: float,
            mean_process_time: float = 0.0,
        ):
        super().__init__()
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.model_weight = model_weight
        self.worker_id = worker_id
        self.fps = fps
        self.mean_process_time = mean_process_time
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.period = self.period - mean_process_time if mean_process_time < self.period else 0

    def run(self):
        model = YoloModel(self.model_weight)
        print(f"[{self.worker_id}] Starting (PID={mp.current_process().pid})", flush=True)
        while True:
            try:
                frame = self.input_queue.get(block=False)
                frame_num = frame[0]
                frame = frame[1]
                predicted = model.process_frame(frame)
                self.result_queue.put((self.worker_id, frame_num, predicted), False)

            except Empty:
                pass
            except Full:
                pass
            if self.stop_event.is_set():
                    break

def apply_mask(image, mask, colored_mask, alpha=0.3):
    """
    Применение цветной маски к изображению

    Args:
        image: исходное изображение
        mask: бинарная маска
        colored_mask: цветная маска
        alpha: прозрачность маски

    Returns:
        image_with_mask: изображение с примененной маской
    """
    # Нормализуем маску
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = mask > 0
    # Применяем маску
    image_with_mask = image.copy()
    image_with_mask[mask] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[mask]
    return image_with_mask

def capture_loop(video_source, input_queue: Queue, res_queue: Queue, stop_event):
    cap_source = video_source
    print("[Capture] Opening video source:", cap_source, flush=True)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {cap_source}")
    to_pil = ToPILImage()

    is_end = False
    frame_idx = 0
    print("[Capture] Starting loop", flush=True)
    try:
        while True:
            success, orig_frame = cap.read()
            if success:
                try:
                    input_queue.put((frame_idx, orig_frame), False)
                    frame_idx += 1
                except Full:
                    pass
            else:
                is_end = True

            try:
                curr = res_queue.get(False)
                proc_id, frame_id, (mask, duration) = curr
                mask = np.array(to_pil(mask.cpu()))
                colored_mask = np.zeros_like(orig_frame)
                colored_mask[:, :] = [0, 255, 0]  # Зеленый цвет

                frame = apply_mask(orig_frame, mask, colored_mask, alpha=0.3)
                cv2.putText(frame, f"Dur.: {(duration * 1000):.0f}ms", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Window", frame)
            except Empty:
                pass
            
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break
            if is_end:
                break
    except KeyboardInterrupt:
        print("[Capture] KeyboardInterrupt caught — exiting capture loop", flush=True)

    finally:
        stop_event.set()
        cap.release()
        print("[Capture] Released video capture device", flush=True)
        cv2.destroyAllWindows()
        print("[Capture] Destroyed all windows", flush=True)

def main():
    input_queue = Queue(1)
    result_queue = Queue(1)
    stop_event = Event()

    worker_ids = ["w1", "w2"]
    worker_fps = [5, 10]
    weights = ["yolo11l-seg.pt", "yolo11l-seg.pt"]

    workers = []
    for id, fps, weight in zip(worker_ids, worker_fps, weights):
        w = FrameProcessor(input_queue, result_queue, stop_event,
                           weight, id, fps)
        workers.append(w)

    for w in workers:
        w.start()
    print("[Main] Workers started", flush=True)

    capture_loop(0, input_queue, result_queue, stop_event)

    try:
        for w in workers:
            w.join()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        print("[Main] Workers joined")

    print("[Main] Shutdown complete.", flush=True)

if __name__ == "__main__":
    main()
