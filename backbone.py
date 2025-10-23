import cv2
import time
import signal
from queue import Empty, Full
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass, field

@dataclass
class Config:
    video_source: int = 0
    max_input_queue_size: int = 1
    worker_configs: list = field(default_factory=lambda: [
        {"worker_id": "worker1", "fps": 5.0, "model_path": "model1.pth"},
        {"worker_id": "worker2", "fps": 2.0, "model_path": "model2.pth"},
    ])
    # add more config options as needed (frame resize, capture throttle, etc.)

class SegmentModel:
    def __init__(self, model_path=None):
        self.model_path = model_path
        # load your actual model here if needed
        # e.g., self.model = load_model(model_path)
        print(f"[SegmentModel] Loaded model from {model_path}")

    def process_frame(self, frame):
        """
        Override this method in subclasses.
        Input: frame (numpy array).
        Output: processed_frame (numpy array).
        """
        return frame

class MySegmentModel(SegmentModel):
    def __init__(self, model_path=None):
        super().__init__(model_path)
        # additional initialization if needed

    def process_frame(self, frame):
        # Example processing: convert to grayscale then back to BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return processed

class FrameProcessor(Process):
    def __init__(
            self,
            input_queue: Queue,
            result_queue: Queue,
            stop_event: Event,
            model: SegmentModel,
            worker_id: str,
            fps: float,
            mean_process_time: float = 0.0,
        ):
        super().__init__()
        self.input_queue = input_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.model = model
        self.worker_id = worker_id
        self.fps = fps
        self.mean_process_time = mean_process_time
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.perion = self.period - mean_process_time if mean_process_time < self.period else 0

    def run(self):
        print(f"[{self.worker_id}] Starting (PID={mp.current_process().pid})", flush=True)
        try:
            while True:
                # Check stop signal
                if self.stop_event.is_set():
                    print(f"[{self.worker_id}] Stop event set — exiting loop", flush=True)
                    break

                try:
                    item = self.input_queue.get(block=False)
                except Empty:
                    # Timeout, loop again (and check stop_event)
                    continue

                if item is None:
                    # Sentinel received
                    print(f"[{self.worker_id}] Received sentinel — exiting", flush=True)
                    break

                frame_index, frame = item
                processed = self.model.process_frame(frame)
                try:
                    self.result_queue.put((self.worker_id, frame_index, processed), block=False)
                except Full:
                    pass

                if self.period > 0:
                    time.sleep(self.period)

        except KeyboardInterrupt:
            print(f"[{self.worker_id}] KeyboardInterrupt caught — exiting", flush=True)

        finally:
            # Signal this worker is done
            self.result_queue.put((self.worker_id, None, None))
            print(f"[{self.worker_id}] Exited", flush=True)

def capture_loop(config: Config, input_queue: Queue, res_queue: Queue, stop_event: Event):
    cap_source = config.video_source
    maxq = config.max_input_queue_size
    print("[Capture] Opening video source:", cap_source, flush=True)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source {cap_source}")

    frame_idx = 0
    print("[Capture] Starting loop", flush=True)
    try:
        while True:
            if stop_event.is_set():
                print("[Capture] Stop event set — exiting capture loop", flush=True)
                break

            ret, frame = cap.read()
            if not ret:
                continue

            if input_queue.qsize() < maxq:
                input_queue.put((frame_idx, frame))
            else:
                print(f"[Capture] Queue full (size={input_queue.qsize()}) — dropping frame {frame_idx}", flush=True)

            frame_idx += 1
            # Optionally you might throttle capture
            # time.sleep(0.001)

            try:
                item = res_queue.get(block=False)
                proc_id, frame_id, frame = item
                if frame is None:
                    break
                cv2.imshow("Window", frame)
                key = cv2.waitKey(10)
                if key == ord("q") or key == 27:
                    break
            except Empty:
                pass



    except KeyboardInterrupt:
        print("[Capture] KeyboardInterrupt caught — exiting capture loop", flush=True)

    finally:
        cap.release()
        print("[Capture] Released video capture device", flush=True)
        # send sentinel values: one per worker
        for _ in config.worker_configs:
            input_queue.put(None)
        print("[Capture] Sent sentinel values to workers", flush=True)

def main():
    config = Config()
    input_queue = Queue(1)
    result_queue = Queue(1)
    stop_event = Event()

    # signal handlers to set stop_event
    def _signal_handler(signum, frame):
        print(f"[Main] Received signal {signum} — initiating shutdown", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Instantiate workers
    workers = []
    for wc in config.worker_configs:
        model = MySegmentModel(model_path=wc["model_path"])
        w = FrameProcessor(input_queue, result_queue, stop_event,
                           model, wc["worker_id"], wc["fps"])
        workers.append(w)

    # Start workers
    for w in workers:
        w.start()
    print("[Main] Workers started", flush=True)

    # Run capture loop in main thread
    capture_loop(config, input_queue, result_queue, stop_event)

    # Wait for workers to finish
    for w in workers:
        w.join()
    print("[Main] Workers joined", flush=True)

    # Process result queue
    print("[Main] Draining result queue", flush=True)
    while not result_queue.empty():
        worker_id, frame_idx, processed_frame = result_queue.get()
        if frame_idx is None:
            print(f"[Main] {worker_id} signalled done", flush=True)
        else:
            print(f"[Main] Got processed frame {frame_idx} from {worker_id}", flush=True)
            cv2.imshow(f"{worker_id}", processed_frame)
            cv2.waitKey(20)

    print("[Main] Shutdown complete.", flush=True)

if __name__ == "__main__":
    main()
