Ссылка на коллаб: https://colab.research.google.com/drive/1ULMwvLFWC6qBxvXjgQt22_hHE4WLVL-0?usp=sharing 

# Детекция и подсчёт транспортных средств на видео с YOLOv8

**Студент:** Морозова Юлия Андреевна  
**Дисциплина:** Компьютерное зрение  
**Преподаватель:** Ярослав Найчук  
**Дата:** 23.05.2026

---

## Описание

Система автоматического подсчёта транспортных средств, проезжающих через заданную линию на видеозаписи. Реализована детекция ТС с помощью YOLOv8, трекинг объектов через встроенный ByteTrack и логика подсчёта пересечений контрольной линии.

## Быстрый старт (Google Colab)

```python
!pip install ultralytics opencv-python numpy lap --quiet
```

Настройте параметры в ячейке «Настройка параметров»:

```python
direction = 'top_to_bottom'       # или 'left_to_right'
line_coords = "0,800,0,800"       # для top_to_bottom: 0,Y,0,Y
target_fps = 10                   # FPS обработки (рекомендуется 10–15)
target_width, target_height = 1920, 1080
conf, iou = 0.5, 0.5
```

Запустите все ячейки — результат сохранится в `output_video.mp4`.

---

## Как это работает

### 1. Детекция (YOLOv8n)
Используется модель `yolov8n.pt`. Из всех детекций отбираются только транспортные средства:

```python
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
mask = np.isin(boxes.cls.cpu().numpy(), vehicle_classes)
```

### 2. Трекинг (ByteTrack)
Трекинг реализован через встроенный `model.track()` из Ultralytics — автоматически назначает каждому объекту уникальный ID:

```python
results = model.track(frame, persist=True, conf=conf, iou=iou, verbose=False)
```

### 3. Линия подсчёта
Линия задаётся через `line_coords` и растягивается на весь кадр:

```python
# top_to_bottom: горизонтальная линия по Y-координате
line_y = pt1_in[1]
line_pt1, line_pt2 = (0, line_y), (target_width, line_y)
```

### 4. Подсчёт пересечений
Проверка пересечения вектора движения объекта с линией через CCW-алгоритм:

```python
def line_intersection(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
```

Каждый объект считается только один раз (через `counted_ids`):

```python
if cross_ok and obj_id not in counted_ids:
    total_crossings += 1
    counted_ids.add(obj_id)
```

### 5. Визуализация
На каждый кадр наносятся:
- Bounding box + класс + ID объекта
- Жёлтая точка в центре (центроид)
- Красная линия подсчёта
- Счётчик пересечений в левом верхнем углу

---

## Результаты

| Параметр | Значение |
|---|---|
| Модель | YOLOv8n (`yolov8n.pt`) |
| Классы ТС | car (2), motorcycle (3), bus (5), truck (7) |
| FPS обработки | 10 |
| Разрешение | 1920×1080 (FullHD) |
| Направление | top_to_bottom |
| Confidence threshold | 0.5 |
| IoU threshold | 0.5 |

---

## Зависимости

```
ultralytics
opencv-python
numpy
lap
```

---

## Структура проекта

```
├── Morozova_komp_zrenie.ipynb   # основной notebook
└── output_video.mp4             # результат обработки
```
