# Инструкция по установке и использованию библиотеки

## Требования

Python 3.11+, Windows 10+, видеокарта NVIDIA с CUDA (желательно)

## Локальная установка
   1. Установите python 3.11+
   2. Создайте виртуальное окружение
      `python -m venv .venv`
   3. Активируйте виртуальное окружение
      `.venv\Scripts\activate`
   4. Клонируйте репозиторий
      `git clone `
   5. `cd .\log-pdm-lib\`
   6. `python -m pip install pip --upgrade`
   7. Установите requirements
      `pip install -r requirements.txt`
   8. Установите репозиторий в виде библиотеки
      `py -m pip install --upgrade build`
      `py -m build`
      `pip install ./dist/log_pdm_lib-1.0.tar.gz`
