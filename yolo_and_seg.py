import cv2
from ultralytics import YOLO
import numpy as np
from scipy.spatial.distance import cosine
import torch


class HumanDetector:
    def __init__(self, detection_model_path='yolov8n.pt', segmentation_model_path='yolov8n-seg.pt'):
        """
        Инициализация детектора и сегментатора

        Args:
            detection_model_path (str): путь к модели YOLOv8 для детекции
            segmentation_model_path (str): путь к модели YOLOv8 для сегментации
        """
        self.detection_model = YOLO(detection_model_path)
        self.segmentation_model = YOLO(segmentation_model_path)
        self.class_names = self.detection_model.names
        self.previous_boxes = []  # Сохраняем боксы с предыдущего кадра
        self.previous_masks = []  # Сохраняем маски с предыдущего кадра
        self.previous_frame = None  # Сохраняем предыдущий кадр
        self.previous_features = None  # Сохраняем фичи предыдущего кадра

    def extract_features(self, frame):
        """
        Извлечение признаков из кадра для сравнения

        Args:
            frame: входной кадр

        Returns:
            np.array: вектор признаков
        """
        # Простой метод: используем гистограммы цветов
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])

        # Нормализуем гистограммы
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        # Объединяем в один вектор
        features = np.concatenate([hist_h, hist_s, hist_v])
        return features

    def are_frames_similar(self, frame1, frame2, threshold=0.3):
        """
        Проверка схожести двух кадров

        Args:
            frame1: первый кадр
            frame2: второй кадр
            threshold: порог схожести (меньше = строже)

        Returns:
            bool: True если кадры похожи
        """
        if frame1 is None or frame2 is None:
            return False

        # Извлекаем признаки
        features1 = self.extract_features(frame1)
        features2 = self.extract_features(frame2)

        # Вычисляем косинусное расстояние
        similarity = 1 - cosine(features1, features2)

        return similarity > threshold

    def apply_segmentation_to_frame(self, frame, boxes):
        """
        Применяет сегментацию к обнаруженным людям и возвращает кадр с сегментированными людьми на черном фоне

        Args:
            frame: исходный кадр
            boxes: список bounding boxes

        Returns:
            tuple: (обработанный кадр, список масок)
        """
        # Создаем черный фон
        segmented_frame = np.zeros_like(frame)
        masks = []

        if len(boxes) == 0:
            return segmented_frame, masks

        for box in boxes:
            x1, y1, x2, y2 = box

            # Вырезаем область с человеком
            person_roi = frame[y1:y2, x1:x2]

            if person_roi.size == 0:
                continue

            # Выполняем сегментацию на вырезанной области
            segmentation_results = self.segmentation_model(person_roi, verbose=False)

            for result in segmentation_results:
                if result.masks is not None:
                    seg_boxes = result.boxes
                    seg_masks = result.masks

                    for i, (seg_box, mask_data) in enumerate(zip(seg_boxes, seg_masks.data)):
                        class_id = int(seg_box.cls[0])
                        confidence = seg_box.conf[0]

                        if class_id == 0 and confidence > 0.5:
                            # Получаем маску
                            mask = mask_data.cpu().numpy()
                            mask = mask[0] if mask.ndim > 2 else mask

                            # Ресайзим маску к размеру вырезанной области
                            mask_resized = cv2.resize(mask, (person_roi.shape[1], person_roi.shape[0]))

                            # Создаем бинарную маску
                            binary_mask = (mask_resized > 0.5).astype(np.uint8)

                            # Применяем маску к вырезанной области
                            person_segment = cv2.bitwise_and(person_roi, person_roi, mask=binary_mask)

                            # Размещаем сегментированного человека на черном фоне
                            segmented_frame[y1:y2, x1:x2] = cv2.bitwise_or(
                                segmented_frame[y1:y2, x1:x2],
                                person_segment
                            )

                            # Создаем полную маску для всего кадра
                            full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            full_mask[y1:y2, x1:x2] = binary_mask
                            masks.append(full_mask)
                            break

        return segmented_frame, masks

    def detect_and_segment_humans(self, frame, confidence_threshold=0.5):
        """
        Детекция и сегментация людей на кадре

        Args:
            frame: входной кадр
            confidence_threshold: порог уверенности

        Returns:
            tuple: (обработанный кадр, список bounding boxes, список масок)
        """
        current_boxes = []
        current_masks = []

        # Детекция с помощью YOLOv8
        results = self.detection_model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Получаем класс и уверенность
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]

                    # Проверяем, что это человек (класс 0 в COCO) и уверенность выше порога
                    if class_id == 0 and confidence > confidence_threshold:
                        # Получаем координаты bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        current_boxes.append((x1, y1, x2, y2))

        # Если на текущем кадре ничего не обнаружено, но есть предыдущие боксы и маски
        if len(current_boxes) == 0 and len(self.previous_boxes) > 0 and self.previous_frame is not None:
            # Проверяем схожесть с предыдущим кадром
            if self.are_frames_similar(self.previous_frame, frame):
                print("Используем боксы и маски с предыдущего кадра (кадры похожи)")
                current_boxes = self.previous_boxes.copy()
                current_masks = self.previous_masks.copy()
            else:
                print("Кадры не похожи, боксы и маски не сохраняются")

        # Применяем сегментацию к обнаруженным людям
        if len(current_boxes) > 0:
            segmented_frame, current_masks = self.apply_segmentation_to_frame(frame, current_boxes)
            print(1)
        else:
            segmented_frame = np.zeros_like(frame)

        # Обновляем предыдущие данные
        self.previous_boxes = current_boxes.copy()
        self.previous_masks = current_masks.copy()
        self.previous_frame = frame.copy()

        return segmented_frame, current_boxes, current_masks

    def process_video(self, input_path, output_path, confidence_threshold=0.5, similarity_threshold=0.3):
        """
        Обработка видеофайла с детекцией и сегментацией

        Args:
            input_path: путь к входному видео
            output_path: путь для сохранения результата
            confidence_threshold: порог уверенности
            similarity_threshold: порог схожести кадров
        """
        # Сбрасываем предыдущие данные
        self.previous_boxes = []
        self.previous_masks = []
        self.previous_frame = None

        # Открываем видео
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print("Ошибка открытия видео файла!")
            return

        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Создаем VideoWriter для сохранения результата
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Обработка видео с сегментацией...")

        frame_count = 0
        boxes_history = []  # История боксов для анализа
        masks_history = []  # История масок для анализа

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция и сегментация людей
            processed_frame, boxes, masks = self.detect_and_segment_humans(frame, confidence_threshold)
            boxes_history.append(len(boxes))
            masks_history.append(len(masks))

            # Добавляем информацию о количестве обнаруженных людей
            cv2.putText(processed_frame, f'People: {len(boxes)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, f'Masks: {len(masks)}',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, f'Frame: {frame_count}',
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Сохраняем кадр
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}")

        # Освобождаем ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Статистика
        frames_with_detection = sum(1 for x in boxes_history if x > 0)
        frames_with_segmentation = sum(1 for x in masks_history if x > 0)
        print(f"Обработка завершена!")
        print(f"Всего кадров: {frame_count}")
        print(f"Кадры с детекцией: {frames_with_detection}")
        print(f"Кадры с сегментацией: {frames_with_segmentation}")
        print(f"Процент успешной детекции: {frames_with_detection / frame_count * 100:.1f}%")
        print(f"Процент успешной сегментации: {frames_with_segmentation / frame_count * 100:.1f}%")
        print(f"Результат сохранен в: {output_path}")


    def process_image(self, input_path, output_path, confidence_threshold=0.5):
        """
        Обработка изображения с детекцией и сегментацией людей

        Args:
            input_path: путь к входному изображению
            output_path: путь для сохранения результата
            confidence_threshold: порог уверенности
        """
        # Загружаем изображение
        image = cv2.imread(input_path)
        if image is None:
            print("Ошибка загрузки изображения!")
            return

        print("Обработка изображения с сегментацией...")

        # Детекция и сегментация людей
        segmented_image, boxes, masks = self.detect_and_segment_humans(image, confidence_threshold)

        # Добавляем информацию о количестве обнаруженных людей
        cv2.putText(segmented_image, f'People: {len(boxes)}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(segmented_image, f'Masks: {len(masks)}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Сохраняем результат
        cv2.imwrite(output_path, segmented_image)

        print(f"Обработка завершена!")
        print(f"Обнаружено людей: {len(boxes)}")
        print(f"Создано масок: {len(masks)}")
        print(f"Результат сохранен в: {output_path}")

        return segmented_image, boxes, masks

    def process_webcam(self, confidence_threshold=0.5):
        """
        Обработка видео с вебкамеры в реальном времени

        Args:
            confidence_threshold: порог уверенности
        """
        # Сбрасываем предыдущие данные
        self.previous_boxes = []
        self.previous_masks = []
        self.previous_frame = None

        # Открываем вебкамеру
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Ошибка открытия вебкамеры!")
            return

        print("Запуск обработки с вебкамеры...")
        print("Нажмите 'q' для выхода")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения кадра с вебкамеры!")
                break

            # Детекция и сегментация людей
            processed_frame, boxes, masks = self.detect_and_segment_humans(frame, confidence_threshold)

            # Добавляем информацию о количестве обнаруженных людей
            cv2.putText(processed_frame, f'People: {len(boxes)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, f'Masks: {len(masks)}',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, "Press 'q' to quit",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Показываем результат
            cv2.imshow('Human Detection & Segmentation', processed_frame)

            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()
        print("Обработка с вебкамеры завершена!")


# Пример использования
# Пример использования
if __name__ == "__main__":
    # Создаем детектор с моделями для детекции и сегментации
    detector = HumanDetector('yolov8s.pt', 'yolov8l-seg.pt')

    # Обработка видеофайла
    detector.process_video('gorin.mp4', 'gorin_large.mp4', confidence_threshold=0.6)

    # Обработка изображения (раскомментируйте для использования)
    # detector.process_image('input.jpg', 'output1_yolo.jpg', confidence_threshold=0.5)
    # detector.process_image('input2.png', 'output2_yolo.jpg', confidence_threshold=0.5)
    # detector.process_image('input3.png', 'output3_yolo.jpg', confidence_threshold=0.5)

    # Обработка с вебкамеры (раскомментируйте для использования)
    # detector.process_webcam(confidence_threshold=0.5)