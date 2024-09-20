# Твой персональный тренер

## Основная идея

Инновационное решение для всех, кто учится чему-то новому и не имеет возможности нанять тренера. 
С моим приложением вы можете загрузить видео с правильной техникой выполнения упражнения и видео с вашим выполнением, далее система автоматически даст оценку, насколько правильно вы выполняете упражнение.

## Описание

Этот проект представляет собой веб-приложение на Flask, которое анализирует видеофайлы с использованием предобученной модели *YOLOv8m-pose* для обнаружения ключевых точек (keypoints) и сравнивает их с эталонными видео.

## Особенности

- Загрузка видеофайлов через веб-интерфейс.
- Обнаружение ключевых точек с использованием модели YOLO.
- Синхронизация видео с использованием алгоритма DTW (Dynamic Time Warping).
- Сравнение ключевых точек и вывод оценки.
- Визуализация ключевых точек и оценки на видео.

## Как пользоваться

Необходимо загрузить два фидео формата *.mp4* снятых с одного ракурса на которых присутствует:
- Эталонное выполнение упражнения
- Ваша попытка выполнения этого упражнения

## Навигация по структуре проекта

```bash
├── src - Исходный код проекта
│   ├── MyProject - содержит весь код, не зависящий от модели и данных
│   ├── myproject_data - содержит код для получения наборов данных для исследований
│   └── myproject_models - содержит нейронные сети
│   │   ├── examples - содержит блокноты Jupyter c примерами
│   │   ├── images - содержит необходимые изображения
│   │   └── scripts - содержит основной скрипт запуска
│   │   │   ├── templates — содержит index.html для веб-интерфейса приложения
│   │   │   └── app.py — основной скрипт запуска приложения
```
