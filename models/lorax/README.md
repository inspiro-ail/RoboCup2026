# RVC: Лоракс

Положите сюда обученную inference-модель `voice.pth` с F0 (pitch guidance).
При наличии добавьте соответствующий ей `voice.index` (исходный `added_*.index` тоже подходит).
Без индекса преобразование работает с index_rate=0.

Исходный датасет: `voices Lorax/` (5 минут 13 секунд уникальных записей).
Запустите `venv\Scripts\python.exe train_lorax.py --epochs 100` из папки Lorax-AI.
После завершения обучения и построения индекса скрипт автоматически установит сюда voice.pth и voice.index.
Состояние обучения: `training/lorax/status.json`. Подробности: ../../README.md.
