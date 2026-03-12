# R-car LEGO (stable)

Стабильная зафиксированная версия на сегодня:
- commit: `499d7ca`
- tag: `v2026-03-12-stable`

Основной исходник проекта: `roboshassi.cpp`.

## 1) Требования

- Windows 10/11
- Visual Studio 2022 (C++ toolchain)
- OpenCV 4.12 (в текущей конфигурации ожидается в `C:\OpenCV-cuda\install`)
- Запущенный видеопоток MJPEG от камеры
- HTTP endpoint команд шасси
- Ollama (если используете VLM-подтверждение) и модель `qwen2.5vl:3b`

## 2) Где настраивать адреса

Файл: `roboshassi.cpp`, структура `Config`:
- `mjpeg_url` (видеопоток)
- `cmd_url` (команды шасси)
- `ollama_url` (VLM API)
- `model`

Текущие значения по умолчанию:
- `http://192.168.1.177:5000/video_feed`
- `http://192.168.1.177:5000/cmd`
- `http://127.0.0.1:11434/api/chat`

## 3) Настройка OpenCV пути

Файл: `.vscode/build_debug.cmd`:
- переменная `OPENCV_ROOT` должна указывать на вашу установку OpenCV.

Файл: `.vscode/launch.json`:
- в `PATH` также должен быть путь к `...\opencv...\bin`.

## 4) Сборка

Из VS Code:
- `Terminal -> Run Build Task` (task: `Build USB Cam App`)

Или из терминала:

```powershell
.\.vscode\build_debug.cmd
```

После успешной сборки создается:
- `usb_cam_app.exe`

Примечание: сообщение `Debug OpenCV lib not found. Using Release lib.` допустимо.

## 5) Запуск

```powershell
.\usb_cam_app.exe
```

## 6) Управление

- `Space` - AUTO ON/OFF
- `W` - вперед
- `X` - назад
- `A` / `D` / `Z` / `C` - повороты
- `M` - переключение MANUAL режима (`MOMENTARY`/`LATCH`)
- `1` или `S` - стоп
- `Q` или `Esc` - выход

## 7) Как запустить именно стабильную версию из Git

```powershell
git clone https://github.com/MaxFrolov239/R-car-LEGO.git
cd R-car-LEGO
git checkout tags/v2026-03-12-stable -b stable-2026-03-12
.\.vscode\build_debug.cmd
.\usb_cam_app.exe
```

## 8) Что смотреть в логах

- `target=yes/no`, `cand`, `src`, `conf`, `area`, `off`
- `decision=...`
- `goalLatched=1` + `decision=goal:latched` означает, что цель считается достигнутой и шасси удерживается в стопе.
- `frameAge=...` показывает задержку кадра.

## 9) Если что-то не работает

- AUTO не едет, только крутится:
  - проверьте `mjpeg_url` и качество/освещение кадра;
  - проверьте, что в кадре реально виден мяч.
- Команды не доходят:
  - проверьте `cmd_url` и HTTP доступность endpoint.
- VLM не подтверждает (`aiSeen=0`, долгий `ageMs`):
  - проверьте `ollama_url`, запущен ли Ollama, и загружена ли модель.

