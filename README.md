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

## 10) Типовые рабочие пороги (ваша камера/свет)

Ниже стартовый профиль, который использован в текущей стабильной версии для вашей сцены (желтый мяч, комнатный свет, низкий ракурс).
Все параметры задаются в `Config` внутри `roboshassi.cpp`.

```cpp
// Fast color/shape detect
fast_h_low=22, fast_h_high=32
fast_s_low=120, fast_s_high=255
fast_v_low=100, fast_v_high=255
fast_min_area_ratio=0.0025
fast_min_circularity=0.42
fast_min_fill_ratio=0.35
fast_min_aspect=0.55
fast_max_aspect=1.85

// Fast lock/move gates
fast_lock_conf_min=0.60
fast_lock_area_min=0.010
fast_move_conf_min=0.64
fast_move_area_min=0.015
fast_no_ball_override_conf=0.60
fast_no_ball_override_area=0.030
vlm_negative_hold_ms=900

// Turn/near behavior
close_area_ratio=0.07
turn_deadband_close=0.22
turn_exit_close=0.16
turn_strong_close=0.48
turn_flip_close=0.36

// Goal and release
goal_area_ratio=0.33
goal_near_area_ratio=0.12
goal_near_center_offset=0.20
goal_near_conf_min=0.78
goal_near_ms=220
goal_release_lost_frames=8

// Reacquire/search stability
target_unlock_frames=30
target_stale_ms=2600
reacquire_hold_ms=700
reacquire_min_area=0.045
reacquire_max_abs_offset=0.30
search_pulse_ms=220
search_hold_ms=380
```

Быстрые корректировки под типовые проблемы:
- Сильный дребезг `rot_l/rot_r` возле мяча:
  - увеличить `turn_deadband_close` на `+0.02..+0.04`;
  - увеличить `turn_flip_close` на `+0.02..+0.04`.
- Рано считает цель достигнутой:
  - увеличить `goal_area_ratio` до `0.35..0.38`;
  - увеличить `goal_near_area_ratio` до `0.13..0.15`.
- Плохо видит мяч в темноте:
  - снизить `fast_lock_conf_min` до `0.55`;
  - снизить `fast_move_conf_min` до `0.58`;
  - снизить `fast_v_low` до `90`.
- Ловит ложные желтые объекты (мебель/пол):
  - поднять `fast_min_circularity` до `0.46..0.50`;
  - поднять `fast_min_fill_ratio` до `0.40..0.45`;
  - поднять `fast_min_area_ratio` до `0.0030..0.0040`.

После изменения порогов:
1. Пересоберите: `.\.vscode\build_debug.cmd`
2. Перезапустите: `.\usb_cam_app.exe`
