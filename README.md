# AITH2026 Speech Recognition and Generation
Репозиторий с домашними заданиями по курсу разпознавания и генерации речи.

Студент: Попов Игорь Владленович


### Домашняя работа №1 Digital Signal Processing
Описание задания: [ссылка](https://github.com/AntonOkhotnikov/ai-talent-hub-itmo-speech-course/tree/main/assignments/assignment1)

#### Get started:
* Установка зависимостей: `pip install -r requirements.txt`
* Скачать датасет SPEECHCOMMANDS: `python scripts/download_speech_comands.py`
* Тестирование LogMelFilterBanks: `python -m pytest`
* Запуск обучения: `python hw_1/main.py`

[PDF Report](assets/hw_1/hw_2_AITH_SR.pdf) (`assets/hw_1/hw_2_AITH_SR.pdf`)


### Домашняя работа №2 ASR Decoding
Описание задания: [ссылка](https://github.com/AntonOkhotnikov/ai-talent-hub-itmo-speech-course/tree/main/assignments/assignment2)

#### Get started:
* Установка зависимостей: `pip install -r requirements_hw2.txt`
* Нужно установить KenLM (см описание задания)
* Скачать модель [4-gram.arpa.gz](https://www.openslr.org/11/) и поместить ее в `weights/4-gram.arpa.gz`
* Запуск кода тасков 1-9: `python hw_2/evaluate.py`
* Имплементация WavDecoder: `hw_2/wav2vec2decoder.py`

[PDF Report](assets/hw_2/hw_2_AITH_SR.pdf) (`assets/hw_2/hw_2_AITH_SR.pdf`)
