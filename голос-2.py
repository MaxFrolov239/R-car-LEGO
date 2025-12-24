# -*- coding: utf-8 -*-
import os, time, cv2, requests, threading, json, base64
import numpy as np
import pyttsx3

# --- НАСТРОЙКИ ---
PI_IP = "192.168.1.177"
MJPEG_URL = f"http://{PI_IP}:5000/video_feed"
CMD_URL   = f"http://{PI_IP}:5000/cmd"

BASE_PAUSE = 0.9      
PULSE_DURATION = 0.12 
YELLOW_LOW = np.array([22, 120, 100])
YELLOW_HIGH = np.array([32, 255, 255])

class VoiceReporter:
    def __init__(self):
        self.last_phrase = ""
        self.busy = False 

    def say(self, text):
        # Если фраза та же или мы еще говорим — выходим
        if text == self.last_phrase or self.busy:
            return

        def _speak(phrase):
            self.busy = True
            try:
                # Инициализируем движок прямо в потоке
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)
                engine.say(phrase)
                engine.runAndWait()
                engine.stop()
                del engine 
            except: 
                pass
            finally:
                time.sleep(1.0) # Пауза перед следующей фразой
                self.busy = False

        self.last_phrase = text
        threading.Thread(target=_speak, args=(text,), daemon=True).start()

class FreshFrame:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
    def _update(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock: self.frame = f.copy()
            else:
                time.sleep(1); self.cap = cv2.VideoCapture(self.url)
    def get(self):
        with self.lock: return (self.frame is not None), self.frame

class RobotLogic:
    def __init__(self, voice):
        self.voice = voice
        self.last_move = 0
        self.ai_confirmed = False
        self.status = "IDLE"
        self.search_count = 0
        self.current_pause = BASE_PAUSE

    def ask_llm(self, frame):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key: return
        try:
            _, buf = cv2.imencode(".jpg", cv2.resize(frame, (320, 240)), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            b64 = base64.b64encode(buf).decode('utf-8')
            r = requests.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Is there a yellow ball here? Answer JSON {'seen':bool}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}], "max_tokens": 50}, timeout=5.0)
            res = json.loads(r.json()['choices'][0]['message']['content'].replace("```json", "").replace("```", ""))
            self.ai_confirmed = res.get("seen", False)
        except: 
            pass

    def move(self, action, pulses=1):
        try:
            self.current_pause = BASE_PAUSE + (0.2 if pulses > 1 else 0)
            for _ in range(pulses):
                requests.get(CMD_URL, params={"a": action}, timeout=0.8)
                time.sleep(PULSE_DURATION)
                requests.get(CMD_URL, params={"a": "stop"}, timeout=0.8)
            self.last_move = time.monotonic()
        except: 
            pass

def main():
    voice = VoiceReporter()
    stream = FreshFrame(MJPEG_URL)
    bot = RobotLogic(voice)
    last_ai_time = 0
    voice.say("Система стабилизирована. Ищу мяч.")

    while True:
        ret, frame = stream.get()
        if not ret: continue

        # Запрос к ИИ
        if time.monotonic() - last_ai_time > 4.0:
            threading.Thread(target=bot.ask_llm, args=(frame.copy(),), daemon=True).start()
            last_ai_time = time.monotonic()

        # Пауза для стабилизации
        if time.monotonic() - bot.last_move < bot.current_pause:
            cv2.imshow("Vision", frame)
            if cv2.waitKey(1) == 27: break
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_found = False
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 1800:
                target_found = True
                cx = int(M["m10"] / M["m00"])
                area = cv2.contourArea(c) / (frame.shape[0] * frame.shape[1])
                pct = cx / frame.shape[1]
                cv2.circle(frame, (cx, int(M["m01"]/M["m00"])), 30, (0, 255, 0), 2)

                if pct < 0.38:
                    bot.status = "ADJUST L"; bot.move("rot_l")
                elif pct > 0.62:
                    bot.status = "ADJUST R"; bot.move("rot_r")
                else:
                    if bot.ai_confirmed:
                        if area < 0.12:
                            bot.status = "FAST FWD"; bot.move("fwd", pulses=3)
                        elif area < 0.38:
                            bot.status = "SLOW FWD"; bot.move("fwd", pulses=1)
                        else:
                            bot.status = "GOAL"
                            voice.say("Вижу мяч вплотную.")
                            requests.get(CMD_URL, params={"a": "stop"})
                    else:
                        bot.status = "VERIFYING AI"
                        voice.say("Вижу объект. Проверяю.")
                        requests.get(CMD_URL, params={"a": "stop"})
            
        if not target_found:
            # Сбрасываем подтверждение ИИ, если OpenCV ничего не видит
            bot.ai_confirmed = False
            bot.search_count += 1
            direction = "rot_r" if (bot.search_count // 40) % 2 == 0 else "rot_l"
            bot.status = f"SCAN {direction}"
            bot.move(direction)

        # Визуализация
        color = (0, 255, 0) if bot.ai_confirmed else (0, 0, 255)
        cv2.putText(frame, f"ST: {bot.status}", (10, 30), 1, 1.3, (255, 255, 255), 2)
        cv2.putText(frame, f"AI: {bot.ai_confirmed}", (10, 65), 1, 1.3, color, 2)
        cv2.imshow("Vision", frame)
        if cv2.waitKey(1) == 27: break

    requests.get(CMD_URL, params={"a": "stop"}); stream.running = False; cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()