#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#endif

using Clock = std::chrono::steady_clock;

template <typename T>
T clamp_value(T value, T lo, T hi) {
    return std::max(lo, std::min(value, hi));
}

int64_t now_ms() {
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count()
    );
}

double mean_absdiff_gray(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 255.0;
    }
    cv::Mat d;
    cv::absdiff(a, b, d);
    return cv::mean(d)[0];
}

double sharpness_lap_var(const cv::Mat& bgr) {
    if (bgr.empty()) {
        return 0.0;
    }
    cv::Mat gray, lap;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

std::string base64_encode(const std::vector<uchar>& data) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((data.size() + 2) / 3) * 4);
    for (size_t i = 0; i < data.size(); i += 3) {
        const uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(i + 1 < data.size() ? data[i + 1] : 0) << 8) |
                           (static_cast<uint32_t>(i + 2 < data.size() ? data[i + 2] : 0));
        out.push_back(k[(n >> 18) & 63]);
        out.push_back(k[(n >> 12) & 63]);
        out.push_back(i + 1 < data.size() ? k[(n >> 6) & 63] : '=');
        out.push_back(i + 2 < data.size() ? k[n & 63] : '=');
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out.push_back(c);
    }
    return out;
}

std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool esc = false;
    for (char c : s) {
        if (!esc) {
            if (c == '\\') esc = true;
            else out.push_back(c);
            continue;
        }
        esc = false;
        if (c == 'n') out.push_back('\n');
        else if (c == 'r') out.push_back('\r');
        else if (c == 't') out.push_back('\t');
        else out.push_back(c);
    }
    return out;
}

std::string shorten(const std::string& s, size_t max_len) {
    if (s.size() <= max_len) return s;
    if (max_len < 4) return s.substr(0, max_len);
    return s.substr(0, max_len - 3) + "...";
}

std::string fmt_double(double v, int prec = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

double rect_iou(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0;
    const int union_area = a.area() + b.area() - inter.area();
    if (union_area <= 0) return 0.0;
    return static_cast<double>(inter.area()) / static_cast<double>(union_area);
}

int64_t uptime_ms() {
    static const int64_t t0 = now_ms();
    return now_ms() - t0;
}

void console_log(const std::string& msg) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    std::cout << "[" << std::setw(8) << uptime_ms() << "ms] " << msg << "\n";
}

cv::Mat letterbox_resize(const cv::Mat& src, int out_w, int out_h) {
    if (src.empty() || out_w <= 0 || out_h <= 0) {
        return src;
    }
    cv::Mat canvas(out_h, out_w, src.type(), cv::Scalar::all(0));
    const double sx = static_cast<double>(out_w) / static_cast<double>(src.cols);
    const double sy = static_cast<double>(out_h) / static_cast<double>(src.rows);
    const double s = std::min(sx, sy);
    const int w = std::max(1, static_cast<int>(std::round(src.cols * s)));
    const int h = std::max(1, static_cast<int>(std::round(src.rows * s)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(w, h), 0, 0, (s < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR);
    const int x = (out_w - w) / 2;
    const int y = (out_h - h) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, w, h)));
    return canvas;
}

struct Config {
    std::string mjpeg_url = "http://192.168.1.177:5000/video_feed";
    std::string cmd_url = "http://192.168.1.177:5000/cmd";
    std::string ollama_url = "http://127.0.0.1:11434/api/chat";
    std::string model = "qwen2.5vl:3b";
    int cmd_timeout_ms = 500;
    int vlm_timeout_ms = 7000;
    int cmd_min_interval_ms = 90;
    int scheduler_keepalive_ms = 70;
    int pulse_duration_drive_ms = 130;
    int pulse_pause_drive_ms = 130;
    int pulse_duration_rotate_ms = 80;
    int pulse_pause_rotate_ms = 150;
    int vlm_interval_ms = 250;
    int preview_width = 960;
    int preview_height = 540;
    int lost_frames_to_search = 14;
    int stop_hold_ms = 280;
    int goal_hold_ms = 1200;
    int pregoal_ms = 450;
    int stuck_frames = 32;
    int unstuck_rotate_ms = 700;
    double vlm_conf_min = 0.45;
    double turn_deadband = 0.12;
    double turn_exit = 0.08;
    double turn_strong = 0.35;
    double turn_flip = 0.22;
    double close_area_ratio = 0.07;
    double close_unlock_area_ratio = 0.18;
    double turn_deadband_close = 0.22;
    double turn_exit_close = 0.16;
    double turn_strong_close = 0.48;
    double turn_flip_close = 0.36;
    double goal_area_ratio = 0.33;
    double goal_near_area_ratio = 0.12;
    double goal_near_center_offset = 0.20;
    double goal_near_conf_min = 0.68;
    double goal_near_force_area_ratio = 0.22;
    double goal_near_force_offset = 0.24;
    double stuck_diff = 3.0;
    double hud_font_scale = 0.42;
    double fast_min_area_ratio = 0.0025;
    double fast_min_circularity = 0.42;
    double fast_min_fill_ratio = 0.35;
    double fast_min_aspect = 0.55;
    double fast_max_aspect = 1.85;
    int fast_h_low = 22;
    int fast_s_low = 120;
    int fast_v_low = 100;
    int fast_h_high = 32;
    int fast_s_high = 255;
    int fast_v_high = 255;
    bool verbose_console = true;
    int console_log_interval_ms = 700;
    int hud_event_lines = 4;
    int ai_max_age_ms = 2000;
    int vlm_negative_hold_ms = 900;
    int vlm_target_match_ms = 1800;
    double vlm_target_iou_min = 0.12;
    int fast_forward_grace_ms = 1400;
    double fast_grace_conf_min = 0.72;
    double fast_grace_min_area = 0.05;
    double fast_grace_max_offset = 0.16;
    double fast_lock_conf_min = 0.60;
    double fast_lock_area_min = 0.010;
    double fast_move_conf_min = 0.64;
    double fast_move_area_min = 0.015;
    double fast_no_ball_override_conf = 0.60;
    double fast_no_ball_override_area = 0.030;
    double vlm_bbox_min_area_ratio = 0.003;
    double vlm_bbox_max_area_ratio = 0.70;
    int fast_confirm_max_age_ms = 1200;
    int fast_probe_ms = 220;
    int manual_command_hold_ms = 180;
    bool manual_latch_mode = false;
    bool search_rotate_when_lost = true;
    bool search_single_direction = true;
    bool search_direction_right = true;
    int search_pulse_ms = 220;
    int search_hold_ms = 380;
    int target_lock_frames = 3;
    int target_unlock_frames = 30;
    int target_unlock_frames_close = 70;
    int target_stale_ms = 2600;
    int target_stale_ms_close = 4500;
    int close_blind_hold_lost_frames = 6;
    int reacquire_hold_ms = 700;
    double reacquire_min_area = 0.045;
    double reacquire_max_abs_offset = 0.30;
    int goal_near_ms = 220;
    int goal_release_lost_frames = 8;
    int manual_event_throttle_ms = 120;
};

class MJPEGReader {
public:
    explicit MJPEGReader(std::string url) : url_(std::move(url)) {}
    ~MJPEGReader() { stop(); }

    bool open() {
        if (running_) return true;
        stop_flag_.store(false);
        running_ = true;
        th_ = std::thread(&MJPEGReader::run, this);

        const int64_t deadline = now_ms() + 2500;
        while (now_ms() < deadline) {
            cv::Mat probe;
            int64_t age_ms = -1;
            if (read(probe, &age_ms)) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return false;
    }
    void stop() {
        stop_flag_.store(true);
        if (th_.joinable()) th_.join();
        running_ = false;
    }
    bool read(cv::Mat& frame, int64_t* age_ms = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!has_frame_ || latest_.empty()) return false;
        frame = latest_.clone();
        if (age_ms) *age_ms = now_ms() - latest_ts_ms_;
        return true;
    }
private:
    bool open_capture() {
        if (cap_.isOpened()) return true;
        if (cap_.open(url_, cv::CAP_FFMPEG) || cap_.open(url_, cv::CAP_ANY)) {
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
            return true;
        }
        return false;
    }

    void run() {
        while (!stop_flag_.load()) {
            if (!open_capture()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(120));
                continue;
            }

            cv::Mat f;
            if (cap_.read(f) && !f.empty()) {
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    latest_ = f.clone();
                    latest_ts_ms_ = now_ms();
                    has_frame_ = true;
                }
            } else {
                cap_.release();
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
        }
        cap_.release();
    }

    std::string url_;
    cv::VideoCapture cap_;
    std::mutex mu_;
    cv::Mat latest_;
    int64_t latest_ts_ms_ = 0;
    bool has_frame_ = false;
    std::thread th_;
    std::atomic<bool> stop_flag_{false};
    bool running_ = false;
};

enum class RobotCmd { Stop, Fwd, Back, Left, Right, RotL, RotR };
const char* cmd_name(RobotCmd c) {
    switch (c) {
        case RobotCmd::Stop: return "stop";
        case RobotCmd::Fwd: return "fwd";
        case RobotCmd::Back: return "back";
        case RobotCmd::Left: return "left";
        case RobotCmd::Right: return "right";
        case RobotCmd::RotL: return "rot_l";
        case RobotCmd::RotR: return "rot_r";
    }
    return "stop";
}

#ifdef _WIN32
std::wstring to_w(const std::string& s) {
    if (s.empty()) return L"";
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), &w[0], n);
    return w;
}
#endif

struct HttpResponse {
    bool transport_ok = false;
    int status_code = 0;
    std::string body;
    std::string error;
};

std::string url_encode_component(const std::string& value) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size() * 3);
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hex[(c >> 4) & 0x0F]);
            out.push_back(hex[c & 0x0F]);
        }
    }
    return out;
}

std::string append_query_param(const std::string& base, const std::string& key, const std::string& value) {
    const char sep = (base.find('?') == std::string::npos) ? '?' : '&';
    return base + sep + url_encode_component(key) + "=" + url_encode_component(value);
}

HttpResponse http_request(
    const std::string& method,
    const std::string& url,
    const std::string& body,
    int timeout_ms,
    const std::string& content_type = "application/json"
) {
    HttpResponse resp;
#ifdef _WIN32
    std::smatch m;
    std::regex re(R"(^([a-zA-Z]+)://([^/:]+)(?::([0-9]+))?(.*)$)");
    if (!std::regex_match(url, m, re)) {
        resp.error = "bad_url";
        return resp;
    }
    const bool https = (m[1].str() == "https");
    const std::string host = m[2].str();
    const int port = m[3].matched ? std::stoi(m[3].str()) : (https ? 443 : 80);
    const std::string path = m[4].str().empty() ? "/" : m[4].str();

    HINTERNET hs = WinHttpOpen(L"usb-cam-cpp/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hs) {
        resp.error = "WinHttpOpen";
        return resp;
    }
    WinHttpSetTimeouts(hs, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
    HINTERNET hc = WinHttpConnect(hs, to_w(host).c_str(), static_cast<INTERNET_PORT>(port), 0);
    if (!hc) {
        resp.error = "WinHttpConnect";
        WinHttpCloseHandle(hs);
        return resp;
    }
    HINTERNET hr = WinHttpOpenRequest(hc, to_w(method).c_str(), to_w(path).c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, https ? WINHTTP_FLAG_SECURE : 0);
    if (!hr) {
        resp.error = "WinHttpOpenRequest";
        WinHttpCloseHandle(hc);
        WinHttpCloseHandle(hs);
        return resp;
    }

    const bool has_body = !body.empty();
    const std::wstring headers = has_body ? to_w("Content-Type: " + content_type + "\r\n") : L"";
    const DWORD len = static_cast<DWORD>(body.size());
    if (!WinHttpSendRequest(
            hr,
            has_body ? headers.c_str() : WINHTTP_NO_ADDITIONAL_HEADERS,
            has_body ? static_cast<DWORD>(headers.size()) : 0,
            has_body ? const_cast<char*>(body.data()) : WINHTTP_NO_REQUEST_DATA,
            len,
            len,
            0
        ) || !WinHttpReceiveResponse(hr, nullptr)) {
        resp.error = "send_or_receive";
        WinHttpCloseHandle(hr);
        WinHttpCloseHandle(hc);
        WinHttpCloseHandle(hs);
        return resp;
    }

    DWORD sc = 0;
    DWORD sc_size = sizeof(sc);
    if (WinHttpQueryHeaders(hr, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &sc, &sc_size, WINHTTP_NO_HEADER_INDEX)) {
        resp.status_code = static_cast<int>(sc);
    }

    std::string out;
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(hr, &avail) || avail == 0) break;
        std::string chunk(static_cast<size_t>(avail), '\0');
        DWORD read = 0;
        if (!WinHttpReadData(hr, chunk.data(), avail, &read)) break;
        chunk.resize(read);
        out += chunk;
    }
    resp.transport_ok = true;
    resp.body = std::move(out);
    WinHttpCloseHandle(hr);
    WinHttpCloseHandle(hc);
    WinHttpCloseHandle(hs);
    return resp;
#else
    (void)method;
    (void)url;
    (void)body;
    (void)timeout_ms;
    resp.error = "not_supported";
    return resp;
#endif
}

class RobotCommander {
public:
    RobotCommander(std::string url, int min_interval_ms, int timeout_ms) : url_(std::move(url)), min_interval_ms_(min_interval_ms), timeout_ms_(timeout_ms) {}
    bool send(RobotCmd cmd, bool force = false) {
        const int64_t t = now_ms();
        if (!force && cmd == last_cmd_ && (t - last_send_ms_) < min_interval_ms_) return true;

        const std::string c = cmd_name(cmd);
        const std::vector<std::string> variants = { append_query_param(url_, "a", c) };

        bool ok = false;
        std::string summary;
        summary.reserve(160);
        std::string used_url;
        int used_status = 0;

        for (const std::string& request_url : variants) {
            const HttpResponse r = http_request("GET", request_url, "", timeout_ms_);
            const bool this_ok = r.transport_ok && r.status_code >= 200 && r.status_code < 300;
            ok = ok || this_ok;

            if (this_ok && used_url.empty()) {
                used_url = request_url;
                used_status = r.status_code;
            } else if (used_url.empty()) {
                used_url = request_url;
                used_status = r.status_code;
            }

            if (!summary.empty()) summary += " ";
            summary += "GET:" + std::to_string(r.status_code);
            if (!r.error.empty()) summary += "(" + r.error + ")";
        }

        last_status_code_ = used_status;
        last_url_ = used_url;
        last_error_ = summary;

        if (ok) {
            last_cmd_ = cmd;
            last_send_ms_ = t;
        }
        return ok;
    }

    int last_status_code() const { return last_status_code_; }
    const std::string& last_url() const { return last_url_; }
    const std::string& last_error() const { return last_error_; }

private:
    std::string url_;
    int min_interval_ms_;
    int timeout_ms_;
    RobotCmd last_cmd_ = RobotCmd::Stop;
    int64_t last_send_ms_ = 0;
    int last_status_code_ = 0;
    std::string last_url_;
    std::string last_error_;
};

class MotionScheduler {
public:
    MotionScheduler(RobotCommander& commander,
                    int keepalive_ms,
                    int pulse_duration_drive_ms,
                    int pulse_pause_drive_ms,
                    int pulse_duration_rotate_ms,
                    int pulse_pause_rotate_ms)
        : commander_(commander),
          keepalive_ms_(keepalive_ms),
          pulse_duration_drive_ms_(pulse_duration_drive_ms),
          pulse_pause_drive_ms_(pulse_pause_drive_ms),
          pulse_duration_rotate_ms_(pulse_duration_rotate_ms),
          pulse_pause_rotate_ms_(pulse_pause_rotate_ms) {}
    void set_desired(RobotCmd c) { desired_ = c; }
    void hold_stop_for(int ms) {
        hold_until_ = std::max(hold_until_, now_ms() + ms);
        desired_ = RobotCmd::Stop;
        in_pulse_ = false;
    }
    void tick() {
        const int64_t t = now_ms();
        const RobotCmd out = (t < hold_until_) ? RobotCmd::Stop : desired_;
        const int pulse_duration_ms = pulse_duration_for(out);
        const int pulse_pause_ms = pulse_pause_for(out);

        if (in_pulse_ && out != pulse_cmd_) {
            if (sent_ != RobotCmd::Stop && commander_.send(RobotCmd::Stop, true)) {
                sent_ = RobotCmd::Stop;
                last_tick_send_ = t;
            }
            in_pulse_ = false;
            next_pulse_at_ = t;
        }

        const bool pulse_enabled = (out != RobotCmd::Stop && pulse_duration_ms > 0 && pulse_pause_ms >= 0);
        if (!pulse_enabled) {
            in_pulse_ = false;
            if (out != sent_ || (t - last_tick_send_) > keepalive_ms_) {
                if (commander_.send(out, true)) {
                    sent_ = out;
                    last_tick_send_ = t;
                }
            }
            return;
        }

        if (out == RobotCmd::Stop) {
            in_pulse_ = false;
            next_pulse_at_ = t;
            if (sent_ != RobotCmd::Stop || (t - last_tick_send_) > keepalive_ms_) {
                if (commander_.send(RobotCmd::Stop, true)) {
                    sent_ = RobotCmd::Stop;
                    last_tick_send_ = t;
                }
            }
            return;
        }

        if (in_pulse_ && t >= pulse_stop_at_) {
            if (commander_.send(RobotCmd::Stop, true)) {
                sent_ = RobotCmd::Stop;
                last_tick_send_ = t;
            }
            in_pulse_ = false;
            next_pulse_at_ = t + pulse_pause_ms;
            return;
        }

        if (!in_pulse_ && t >= next_pulse_at_) {
            if (commander_.send(out, true)) {
                sent_ = out;
                last_tick_send_ = t;
                pulse_cmd_ = out;
            }
            in_pulse_ = true;
            pulse_stop_at_ = t + pulse_duration_ms;
        }
    }
    RobotCmd sent_cmd() const { return sent_; }
    RobotCmd desired_cmd() const { return desired_; }
    int64_t hold_remaining_ms(int64_t t) const { return std::max<int64_t>(0, hold_until_ - t); }
    int last_status_code() const { return commander_.last_status_code(); }
    const std::string& last_url() const { return commander_.last_url(); }
    const std::string& last_error() const { return commander_.last_error(); }
private:
    static bool is_rotate_cmd(RobotCmd c) {
        return c == RobotCmd::RotL || c == RobotCmd::RotR || c == RobotCmd::Left || c == RobotCmd::Right;
    }
    int pulse_duration_for(RobotCmd c) const {
        return is_rotate_cmd(c) ? pulse_duration_rotate_ms_ : pulse_duration_drive_ms_;
    }
    int pulse_pause_for(RobotCmd c) const {
        return is_rotate_cmd(c) ? pulse_pause_rotate_ms_ : pulse_pause_drive_ms_;
    }

    RobotCommander& commander_;
    int keepalive_ms_;
    int pulse_duration_drive_ms_;
    int pulse_pause_drive_ms_;
    int pulse_duration_rotate_ms_;
    int pulse_pause_rotate_ms_;
    RobotCmd desired_ = RobotCmd::Stop;
    RobotCmd sent_ = RobotCmd::Stop;
    RobotCmd pulse_cmd_ = RobotCmd::Stop;
    int64_t hold_until_ = 0;
    int64_t last_tick_send_ = 0;
    bool in_pulse_ = false;
    int64_t pulse_stop_at_ = 0;
    int64_t next_pulse_at_ = 0;
};

struct Detection {
    bool valid = false;
    bool ball = false;
    cv::Rect bbox;
    double conf = 0.0;
    std::string source;
};

std::optional<Detection> parse_detection(const std::string& text, cv::Size frame) {
    std::smatch m;
    std::regex c("\"content\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
    std::string payload = text;
    if (std::regex_search(text, m, c)) payload = json_unescape(m[1].str());

    Detection d;
    d.valid = true;
    auto parse_bool_token = [](std::string token, bool& value) -> bool {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), token.end());
        if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
            token = token.substr(1, token.size() - 2);
        }
        std::string norm;
        norm.reserve(token.size());
        for (char ch : token) {
            norm.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        if (norm == "true" || norm == "1") {
            value = true;
            return true;
        }
        if (norm == "false" || norm == "0") {
            value = false;
            return true;
        }
        return false;
    };

    std::regex ball_re(R"("ball"\s*:\s*(true|false|1|0|"true"|"false"|"1"|"0"))", std::regex::icase);
    std::regex seen_re(R"("seen"\s*:\s*(true|false|1|0|"true"|"false"|"1"|"0"))", std::regex::icase);
    bool parsed = false;
    bool parsed_value = false;
    if (std::regex_search(payload, m, ball_re) && parse_bool_token(m[1].str(), parsed_value)) {
        d.ball = parsed_value;
        parsed = true;
    } else if (std::regex_search(payload, m, seen_re) && parse_bool_token(m[1].str(), parsed_value)) {
        d.ball = parsed_value;
        d.source = "vlm_seen";
        parsed = true;
    }
    if (!parsed) {
        return std::nullopt;
    }

    std::regex conf_re(R"("conf"\s*:\s*([-+]?[0-9]*\.?[0-9]+))");
    if (std::regex_search(payload, m, conf_re)) d.conf = std::stod(m[1].str());
    else d.conf = d.ball ? 0.7 : 0.0;
    if (!d.ball) return d;

    std::regex bbox_re(R"("bbox"\s*:\s*\[\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)\s*\])");
    if (!std::regex_search(payload, m, bbox_re)) {
        if (d.source.empty()) d.source = "vlm_seen";
        return d; // allow "seen:true" without bbox (Python-style AI gate)
    }
    double x = std::stod(m[1].str()), y = std::stod(m[2].str()), w = std::stod(m[3].str()), h = std::stod(m[4].str());
    if (w <= 1.5 && h <= 1.5) { x *= frame.width; y *= frame.height; w *= frame.width; h *= frame.height; }
    d.bbox = cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h)) & cv::Rect(0, 0, frame.width, frame.height);
    d.source = "vlm";
    return d;
}

class VLMClient {
public:
    VLMClient(std::string endpoint, std::string model, int timeout) : endpoint_(std::move(endpoint)), model_(std::move(model)), timeout_(timeout) {}
    std::optional<Detection> infer(const cv::Mat& frame) const {
        std::vector<uchar> jpg;
        if (!cv::imencode(".jpg", frame, jpg, {cv::IMWRITE_JPEG_QUALITY, 82})) return std::nullopt;
        std::ostringstream body;
        body << "{\"model\":\"" << json_escape(model_) << "\",\"stream\":false,\"format\":\"json\",\"messages\":[{\"role\":\"user\",\"content\":\"Return strict JSON only. Prefer {\\\"ball\\\":true|false,\\\"bbox\\\":[x,y,w,h],\\\"conf\\\":0..1}. If unsure bbox, return {\\\"seen\\\":true|false}.\",\"images\":[\"" << base64_encode(jpg) << "\"]}]}";
        const HttpResponse resp = http_request("POST", endpoint_, body.str(), timeout_);
        if (!resp.transport_ok || resp.status_code < 200 || resp.status_code >= 300) return std::nullopt;
        return parse_detection(resp.body, frame.size());
    }
private:
    std::string endpoint_;
    std::string model_;
    int timeout_;
};

class AsyncVLMWorker {
public:
    AsyncVLMWorker(VLMClient client, int period_ms) : client_(std::move(client)), period_ms_(period_ms), th_(&AsyncVLMWorker::run, this) {}
    ~AsyncVLMWorker() { { std::lock_guard<std::mutex> lk(mu_); stop_ = true; } cv_.notify_one(); if (th_.joinable()) th_.join(); }
    void submit(const cv::Mat& frame) { std::lock_guard<std::mutex> lk(mu_); pending_ = frame.clone(); has_pending_ = true; cv_.notify_one(); }
    bool try_pop(Detection& d) { std::lock_guard<std::mutex> lk(mu_); if (!has_out_) return false; d = out_; has_out_ = false; return true; }
private:
    void run() {
        int64_t last = 0;
        while (true) {
            cv::Mat f;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || has_pending_; });
                if (stop_) break;
                const int64_t wait_ms = period_ms_ - (now_ms() - last);
                if (wait_ms > 0) { lk.unlock(); std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms)); lk.lock(); if (stop_) break; }
                f = pending_.clone();
                has_pending_ = false;
            }
            Detection out;
            out.valid = false;
            out.ball = false;
            out.source = "vlm_none";
            const auto r = client_.infer(f);
            last = now_ms();
            if (r.has_value()) {
                out = *r;
            }
            std::lock_guard<std::mutex> lk(mu_);
            out_ = out;
            has_out_ = true;
        }
    }
    VLMClient client_;
    int period_ms_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    cv::Mat pending_;
    bool has_pending_ = false;
    Detection out_;
    bool has_out_ = false;
};

std::optional<Detection> detect_fast_ball(const cv::Mat& frame, const Config& cfg) {
    cv::Mat hsv, mask;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(
        hsv,
        cv::Scalar(cfg.fast_h_low, cfg.fast_s_low, cfg.fast_v_low),
        cv::Scalar(cfg.fast_h_high, cfg.fast_s_high, cfg.fast_v_high),
        mask
    );
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(5, 5, CV_8U));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return std::nullopt;

    size_t best = static_cast<size_t>(-1);
    double best_score = -1.0;
    double best_area = 0.0;

    for (size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area <= 1.0) continue;

        const cv::Rect b = cv::boundingRect(contours[i]);
        if (b.width <= 3 || b.height <= 3) continue;
        const double aspect = static_cast<double>(b.width) / static_cast<double>(b.height);
        if (aspect < cfg.fast_min_aspect || aspect > cfg.fast_max_aspect) continue;

        const double frame_area = static_cast<double>(frame.total());
        const double area_ratio = area / frame_area;
        if (area_ratio < cfg.fast_min_area_ratio) continue;

        const double perimeter = cv::arcLength(contours[i], true);
        if (perimeter <= 1.0) continue;
        const double circularity = (4.0 * CV_PI * area) / (perimeter * perimeter);
        if (circularity < cfg.fast_min_circularity) continue;

        const double fill_ratio = area / static_cast<double>(b.area());
        if (fill_ratio < cfg.fast_min_fill_ratio) continue;

        // Combined score: prefer larger and more circular candidates.
        const double score = area_ratio * (0.6 + 0.4 * clamp_value(circularity, 0.0, 1.0));
        if (score > best_score) {
            best_score = score;
            best = i;
            best_area = area;
        }
    }

    if (best == static_cast<size_t>(-1)) return std::nullopt;

    const auto& best_cnt = contours[best];
    const cv::Rect best_box = cv::boundingRect(best_cnt);
    const double area_ratio = best_area / static_cast<double>(frame.total());
    const double perimeter = std::max(1.0, cv::arcLength(best_cnt, true));
    const double circularity = clamp_value((4.0 * CV_PI * best_area) / (perimeter * perimeter), 0.0, 1.0);
    const double fill_ratio = clamp_value(best_area / std::max(1.0, static_cast<double>(best_box.area())), 0.0, 1.0);
    const double aspect = static_cast<double>(best_box.width) / std::max(1, best_box.height);
    const double aspect_score = clamp_value(1.0 - std::abs(aspect - 1.0) / 0.85, 0.0, 1.0);
    const double area_score = clamp_value(area_ratio / 0.22, 0.0, 1.0);
    const bool touches_edge = (best_box.x <= 2) ||
                              (best_box.y <= 2) ||
                              (best_box.x + best_box.width >= frame.cols - 2) ||
                              (best_box.y + best_box.height >= frame.rows - 2);
    Detection d;
    d.valid = true;
    d.ball = true;
    d.bbox = best_box;
    d.conf = clamp_value(0.10 + 0.42 * circularity + 0.18 * fill_ratio + 0.16 * aspect_score + 0.20 * area_score, 0.10, 0.97);
    if (touches_edge) {
        d.conf *= 0.75;
    }
    d.source = "fast";
    return d;
}

class Navigator {
public:
    Navigator(const Config& cfg, MotionScheduler& motion, AsyncVLMWorker& worker)
        : cfg_(cfg), motion_(motion), worker_(worker) {
        manual_latch_active_ = cfg_.manual_latch_mode;
        search_right_ = cfg_.search_direction_right;
        push_event(std::string("navigator_ready auto=OFF manual=") + (cfg_.manual_latch_mode ? "LATCH" : "MOMENTARY"));
    }

    void set_frame_age_ms(int64_t age_ms) {
        frame_age_ms_ = age_ms;
    }

    void on_key(int key) {
        last_key_ = key;
        const int k = key & 0xFF;
        const bool k_w = (k == 'w' || k == 'W');
        const bool k_x = (k == 'x' || k == 'X');
        const bool k_a = (k == 'a' || k == 'A');
        const bool k_d = (k == 'd' || k == 'D');
        const bool k_z = (k == 'z' || k == 'Z');
        const bool k_c = (k == 'c' || k == 'C');
        const bool k_s = (k == 's' || k == 'S');
        const bool k_q = (k == 'q' || k == 'Q');

        const bool arrow_up = (key == 2490368);
        const bool arrow_down = (key == 2621440);
        const bool arrow_left = (key == 2424832);
        const bool arrow_right = (key == 2555904);

        if (key == ' ') {
            auto_ = !auto_;
            if (!auto_) {
                goal_latched_ = false;
                goal_release_lost_ = 0;
            }
            motion_.hold_stop_for(200);
            push_event(std::string("toggle_auto -> ") + (auto_ ? "ON" : "OFF"));
            return;
        }
        if (k == 'm' || k == 'M') {
            manual_latch_active_ = !manual_latch_active_;
            push_event(std::string("manual_mode -> ") + (manual_latch_active_ ? "LATCH" : "MOMENTARY"));
            return;
        }
        if (k_s || k == '1') {
            manual_ = RobotCmd::Stop;
            motion_.hold_stop_for(200);
            manual_until_ms_ = now_ms();
            push_event("manual_stop");
            return;
        }
        if (k_q) { return; }

        if (!auto_) {
            if (k_w || arrow_up || k == '2') manual_ = RobotCmd::Fwd;
            else if (k_x || arrow_down || k == '3') manual_ = RobotCmd::Back;
            else if (k_a || arrow_left || k == '4' || k_z || k == '6') manual_ = RobotCmd::RotL;
            else if (k_d || arrow_right || k == '5' || k_c || k == '7') manual_ = RobotCmd::RotR;
            else return;
            manual_until_ms_ = now_ms() + cfg_.manual_command_hold_ms;
            if (manual_ != last_manual_event_cmd_ || (now_ms() - last_manual_event_ms_) > cfg_.manual_event_throttle_ms) {
                push_event(std::string("manual_desired=") + cmd_name(manual_) + " key=" + std::to_string(key));
                last_manual_event_ms_ = now_ms();
                last_manual_event_cmd_ = manual_;
            }
        }
    }

    void step(cv::Mat& frame) {
        ++frame_count_;
        const int64_t t = now_ms();
        if (auto_) {
            auto_step(frame, t);
        } else {
            phase_ = "MANUAL";
            if (manual_latch_active_) {
                decision_reason_ = std::string("manual_latch=") + cmd_name(manual_);
                motion_.set_desired(manual_);
            } else if (t <= manual_until_ms_) {
                decision_reason_ = std::string("manual_desired=") + cmd_name(manual_);
                motion_.set_desired(manual_);
            } else {
                decision_reason_ = "manual_idle_timeout";
                motion_.set_desired(RobotCmd::Stop);
            }
        }

        stuck_step(frame, t);
        const RobotCmd before_sent = motion_.sent_cmd();
        const RobotCmd before_desired = motion_.desired_cmd();
        motion_.tick();
        const RobotCmd after_sent = motion_.sent_cmd();

        if (before_desired != motion_.desired_cmd()) {
            push_event(std::string("desired->") + cmd_name(motion_.desired_cmd()));
        }
        if (after_sent != before_sent) {
            push_event(std::string("sent->") + cmd_name(after_sent));
        }

        if (cfg_.verbose_console && (t - last_console_log_ms_) >= cfg_.console_log_interval_ms) {
            console_log(summary_line(t));
            last_console_log_ms_ = t;
        }
        draw(frame, t);
    }

private:
    void push_event(const std::string& e) {
        std::ostringstream oss;
        oss << "t=" << (uptime_ms() / 1000.0) << " " << e;
        events_.push_back(oss.str());
        if (static_cast<int>(events_.size()) > cfg_.hud_event_lines) {
            events_.erase(events_.begin());
        }
        if (cfg_.verbose_console) {
            console_log(e);
        }
    }

    std::string summary_line(int64_t t) const {
        const int64_t vlm_fresh_ms = (last_vlm_rx_ms_ > 0) ? (t - last_vlm_rx_ms_) : -1;
        std::ostringstream oss;
        oss << "phase=" << phase_
            << " auto=" << (auto_ ? "ON" : "OFF")
            << " desired=" << cmd_name(motion_.desired_cmd())
            << " sent=" << cmd_name(motion_.sent_cmd())
            << " holdMs=" << motion_.hold_remaining_ms(t)
            << " target=" << (has_target_ ? "yes" : "no")
            << " cand=" << (has_candidate_ ? std::to_string(candidate_count_) : "0")
            << " src=" << (has_target_ ? target_.source : "none")
            << " conf=" << fmt_double(has_target_ ? target_.conf : 0.0, 2)
            << " aiSeen=" << (ai_confirm_recent_ ? "1" : "0")
            << " vlmFreshMs=" << vlm_fresh_ms
            << " off=" << fmt_double(last_offset_, 3)
            << " area=" << fmt_double(last_area_ratio_, 3)
            << " frameAge=" << frame_age_ms_
            << " lost=" << lost_
            << " turn=" << turn_state_
            << " goalLatched=" << (goal_latched_ ? "1" : "0")
            << " stuck=" << stuck_count_
            << " unlock_reason=" << unlock_reason_
            << " decision=" << decision_reason_;
        return oss.str();
    }

    void auto_step(const cv::Mat& frame, int64_t t) {
        phase_ = "AUTO";
        const double frame_area = static_cast<double>(frame.total());
        ai_confirm_recent_ = (ai_confirm_ms_ > 0) && ((t - ai_confirm_ms_) < cfg_.ai_max_age_ms);
        if ((t - last_submit_) >= cfg_.vlm_interval_ms) {
            worker_.submit(frame);
            last_submit_ = t;
        }
        if (motion_.hold_remaining_ms(t) > 0) {
            motion_.set_desired(RobotCmd::Stop);
            has_candidate_ = false;
            candidate_count_ = 0;
            turn_state_ = 0;
            last_offset_ = 0.0;
            last_area_ratio_ = 0.0;
            decision_reason_ = "hold:cooldown";
            return;
        }
        if (goal_latched_) {
            motion_.set_desired(RobotCmd::Stop);
            Detection vlm;
            if (worker_.try_pop(vlm)) {
                last_vlm_rx_ms_ = t;
                last_vlm_conf_ = vlm.conf;
                const bool vlm_has_bbox = (vlm.bbox.width > 1 && vlm.bbox.height > 1);
                const double vlm_area_ratio = vlm_has_bbox ? (static_cast<double>(vlm.bbox.area()) / frame_area) : 0.0;
                const bool vlm_bbox_sane = vlm_has_bbox &&
                                           vlm_area_ratio >= cfg_.vlm_bbox_min_area_ratio &&
                                           vlm_area_ratio <= cfg_.vlm_bbox_max_area_ratio;
                const bool vlm_ball_strong = vlm.valid && vlm.ball && vlm_bbox_sane && vlm.conf >= cfg_.vlm_conf_min;
                last_vlm_ball_ = vlm_ball_strong;
                if (vlm.valid && !vlm.ball) {
                    ai_confirm_ms_ = 0;
                    ai_confirm_recent_ = false;
                    last_vlm_no_ball_ms_ = t;
                    last_vlm_has_bbox_ = false;
                } else if (vlm_ball_strong) {
                    ai_confirm_ms_ = t;
                    ai_confirm_recent_ = true;
                    last_vlm_ball_ms_ = t;
                    last_vlm_bbox_ = vlm.bbox;
                    last_vlm_has_bbox_ = true;
                }
            }

            const auto fast = detect_fast_ball(frame, cfg_);
            bool close_enough = false;
            if (fast.has_value()) {
                has_target_ = true;
                target_ = *fast;
                const cv::Rect b = target_.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
                if (b.width > 1 && b.height > 1) {
                    const double area = static_cast<double>(b.area()) / frame_area;
                    last_area_ratio_ = area;
                    const double cx = (b.x + b.width * 0.5) / static_cast<double>(frame.cols);
                    last_offset_ = cx - 0.5;
                    close_enough = area >= cfg_.goal_near_area_ratio;
                } else {
                    last_area_ratio_ = 0.0;
                    last_offset_ = 0.0;
                }
            } else {
                has_target_ = false;
                last_area_ratio_ = 0.0;
                last_offset_ = 0.0;
            }
            if (close_enough) {
                goal_release_lost_ = 0;
                decision_reason_ = "goal:latched";
            } else {
                ++goal_release_lost_;
                decision_reason_ = "goal:latched_release_wait";
            }
            if (goal_release_lost_ > cfg_.goal_release_lost_frames) {
                goal_latched_ = false;
                goal_release_lost_ = 0;
                push_event("goal_latch_released");
            }
            return;
        }

        std::optional<Detection> candidate;
        Detection vlm;
        if (worker_.try_pop(vlm)) {
            last_vlm_rx_ms_ = t;
            last_vlm_conf_ = vlm.conf;

            const bool vlm_has_bbox = (vlm.bbox.width > 1 && vlm.bbox.height > 1);
            const double vlm_area_ratio = vlm_has_bbox ? (static_cast<double>(vlm.bbox.area()) / frame_area) : 0.0;
            const bool vlm_bbox_sane = vlm_has_bbox &&
                                       vlm_area_ratio >= cfg_.vlm_bbox_min_area_ratio &&
                                       vlm_area_ratio <= cfg_.vlm_bbox_max_area_ratio;
            const bool vlm_ball_strong = vlm.valid && vlm.ball && vlm_bbox_sane && vlm.conf >= cfg_.vlm_conf_min;
            last_vlm_ball_ = vlm_ball_strong;

            if (vlm.valid && !vlm.ball) {
                ai_confirm_ms_ = 0;
                ai_confirm_recent_ = false;
                last_vlm_no_ball_ms_ = t;
                last_vlm_has_bbox_ = false;
            } else if (vlm_ball_strong) {
                ai_confirm_ms_ = t;
                ai_confirm_recent_ = true;
                last_vlm_ball_ms_ = t;
                last_vlm_bbox_ = vlm.bbox;
                last_vlm_has_bbox_ = true;
            }

            if (vlm_ball_strong) {
                candidate = vlm;
                decision_reason_ = "candidate:vlm_bbox";
            } else if (vlm.valid && vlm.ball && !vlm_bbox_sane) {
                decision_reason_ = "vlm_ball_bbox_outlier";
            } else if (vlm.valid && vlm.ball) {
                decision_reason_ = "vlm_ball_weak";
            } else {
                decision_reason_ = "vlm_reject_or_no_ball";
            }
        }

        if (!candidate.has_value()) {
            const auto fast = detect_fast_ball(frame, cfg_);
            if (fast.has_value()) {
                last_fast_seen_ms_ = t;
                last_fast_conf_ = fast->conf;
                const double fast_area_ratio = static_cast<double>(fast->bbox.area()) / frame_area;
                bool fast_vlm_match = false;
                if (last_vlm_has_bbox_ && last_vlm_ball_ms_ > 0 && ((t - last_vlm_ball_ms_) < cfg_.vlm_target_match_ms)) {
                    const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);
                    const cv::Rect fb = fast->bbox & frame_rect;
                    const cv::Rect vb = last_vlm_bbox_ & frame_rect;
                    if (fb.width > 1 && fb.height > 1 && vb.width > 1 && vb.height > 1) {
                        fast_vlm_match = rect_iou(fb, vb) >= cfg_.vlm_target_iou_min;
                    }
                }

                const bool fast_quality_ok = fast->conf >= cfg_.fast_lock_conf_min &&
                                             fast_area_ratio >= cfg_.fast_lock_area_min;
                if (fast_quality_ok || fast_vlm_match) {
                    candidate = fast;
                    decision_reason_ = fast_vlm_match ? "candidate:fast_vlm_match" : "candidate:fast";
                } else {
                    decision_reason_ = "fast_reject:low_quality";
                }
            }
        }
        if (candidate.has_value() && candidate->source == "fast") {
            const bool vlm_no_ball_recent = (last_vlm_no_ball_ms_ > 0) && ((t - last_vlm_no_ball_ms_) < cfg_.vlm_negative_hold_ms);
            const double cand_area_ratio = static_cast<double>(candidate->bbox.area()) / frame_area;
            const bool fast_no_ball_override = candidate->conf >= cfg_.fast_no_ball_override_conf &&
                                               cand_area_ratio >= cfg_.fast_no_ball_override_area;
            if (vlm_no_ball_recent && !fast_no_ball_override) {
                candidate = std::nullopt;
                decision_reason_ = "fast_reject:vlm_no_ball_recent";
            } else if (vlm_no_ball_recent && fast_no_ball_override) {
                decision_reason_ = "candidate:fast_override_no_ball";
            }
        }

        if (has_target_) {
            if (candidate.has_value()) {
                target_ = candidate.value();
                last_target_ms_ = t;
                lost_ = 0;
                unlock_reason_ = "none";
                has_candidate_ = false;
                candidate_count_ = 0;
                decision_reason_ = std::string("track:update_") + target_.source;
            } else {
                ++lost_;
                const bool close_unlock_zone = last_area_ratio_ >= cfg_.close_unlock_area_ratio;
                const int unlock_frames_limit = close_unlock_zone ? cfg_.target_unlock_frames_close : cfg_.target_unlock_frames;
                const int stale_limit_ms = close_unlock_zone ? cfg_.target_stale_ms_close : cfg_.target_stale_ms;
                const bool by_lost = lost_ > unlock_frames_limit;
                const bool by_stale = (t - last_target_ms_) > stale_limit_ms;
                if (by_lost || by_stale) {
                    if (by_lost && by_stale) unlock_reason_ = close_unlock_zone ? "lost+stale_close" : "lost+stale";
                    else if (by_lost) unlock_reason_ = close_unlock_zone ? "lost_frames_close" : "lost_frames";
                    else unlock_reason_ = close_unlock_zone ? "target_stale_close" : "target_stale";
                    if (last_area_ratio_ >= cfg_.reacquire_min_area &&
                        std::abs(last_offset_) <= cfg_.reacquire_max_abs_offset) {
                        reacquire_hold_until_ms_ = t + cfg_.reacquire_hold_ms;
                    }
                    search_hint_right_ = (last_offset_ >= 0.0);
                    search_hint_valid_ = true;
                    has_target_ = false;
                    push_event("unlock reason=" + unlock_reason_ + " lost=" + std::to_string(lost_) +
                               " ageMs=" + std::to_string(t - last_target_ms_));
                    decision_reason_ = "search:target_unlocked";
                }
            }
        } else {
            if (candidate.has_value()) {
                if (!has_candidate_) {
                    candidate_ = candidate.value();
                    candidate_count_ = 1;
                    has_candidate_ = true;
                } else {
                    const double iou = rect_iou(candidate_->bbox, candidate->bbox);
                    if (candidate_->source == candidate->source || iou > 0.15) {
                        candidate_ = candidate.value();
                        candidate_count_++;
                    } else {
                        candidate_ = candidate.value();
                        candidate_count_ = 1;
                    }
                }

                decision_reason_ = "candidate_wait(" + std::to_string(candidate_count_) + "/" + std::to_string(cfg_.target_lock_frames) + ")";
                if (candidate_count_ >= cfg_.target_lock_frames) {
                    target_ = candidate_.value();
                    has_target_ = true;
                    lost_ = 0;
                    unlock_reason_ = "none";
                    last_target_ms_ = t;
                    has_candidate_ = false;
                    candidate_count_ = 0;
                    push_event(std::string("target_lock src=") + target_.source + " conf=" + fmt_double(target_.conf, 2));
                    decision_reason_ = std::string("track:target_locked_") + target_.source;
                }
            } else {
                has_candidate_ = false;
                candidate_count_ = 0;
                ++lost_;
                decision_reason_ = "search:no_candidate";
            }
        }

        if (has_target_ && target_.source == "fast") {
            if (!fast_target_active_) {
                fast_target_active_ = true;
                fast_target_acquired_ms_ = t;
            }
        } else {
            fast_target_active_ = false;
            fast_target_acquired_ms_ = 0;
        }

        if (!has_target_) {
            if (t < reacquire_hold_until_ms_) {
                motion_.set_desired(RobotCmd::Stop);
                decision_reason_ = "reacquire:hold";
                turn_state_ = 0;
                return;
            }
            if (cfg_.search_rotate_when_lost) {
                if (!search_phase_initialized_) {
                    search_phase_initialized_ = true;
                    search_rotating_ = true;
                    search_phase_until_ms_ = t + cfg_.search_pulse_ms;
                    search_step_ = 0;
                    if (search_hint_valid_) {
                        search_right_ = search_hint_right_;
                        search_hint_valid_ = false;
                    } else if (cfg_.search_single_direction) {
                        search_right_ = cfg_.search_direction_right;
                    }
                    push_event(std::string("search_start dir=") + (search_right_ ? "R" : "L") +
                               (cfg_.search_single_direction ? " mode=single" : " mode=alternate"));
                }
                if (search_rotating_) {
                    if (t < search_phase_until_ms_) {
                        motion_.set_desired(search_right_ ? RobotCmd::RotR : RobotCmd::RotL);
                        decision_reason_ = search_right_ ? "search:pulse_rot_r" : "search:pulse_rot_l";
                    } else {
                        search_rotating_ = false;
                        search_phase_until_ms_ = t + cfg_.search_hold_ms;
                        motion_.set_desired(RobotCmd::Stop);
                        decision_reason_ = "search:analyze_hold";
                    }
                } else {
                    motion_.set_desired(RobotCmd::Stop);
                    decision_reason_ = "search:analyze_hold";
                    if (t >= search_phase_until_ms_) {
                        search_rotating_ = true;
                        if (!cfg_.search_single_direction) {
                            search_right_ = !search_right_;
                        }
                        search_phase_until_ms_ = t + cfg_.search_pulse_ms;
                        search_step_++;
                    }
                }
            } else {
                motion_.set_desired(RobotCmd::Stop);
                decision_reason_ = "search:hold";
            }
            turn_state_ = 0;
            last_offset_ = 0.0;
            last_area_ratio_ = 0.0;
            return;
        }

        search_phase_initialized_ = false;
        reacquire_hold_until_ms_ = 0;

        const cv::Rect b = target_.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (b.width <= 1 || b.height <= 1) {
            has_target_ = false;
            unlock_reason_ = "invalid_bbox";
            push_event("unlock reason=invalid_bbox");
            decision_reason_ = "search:invalid_bbox";
            return;
        }
        const double cx = (b.x + b.width * 0.5) / static_cast<double>(frame.cols);
        ema_x_ = ema_init_ ? (0.88 * ema_x_ + 0.12 * cx) : cx;
        ema_init_ = true;
        const double offset = ema_x_ - 0.5;
        last_offset_ = offset;
        const double area = static_cast<double>(b.area()) / static_cast<double>(frame.total());
        last_area_ratio_ = area;
        const bool close_track = (area >= cfg_.close_area_ratio);
        const double turn_deadband = close_track ? cfg_.turn_deadband_close : cfg_.turn_deadband;
        const double turn_exit = close_track ? cfg_.turn_exit_close : cfg_.turn_exit;
        const double turn_strong = close_track ? cfg_.turn_strong_close : cfg_.turn_strong;
        const double turn_flip = close_track ? cfg_.turn_flip_close : cfg_.turn_flip;
        const double abs_off = std::abs(offset);
        const bool near_goal_base = (area >= cfg_.goal_near_area_ratio) &&
                                    (abs_off <= cfg_.goal_near_center_offset) &&
                                    (target_.conf >= cfg_.goal_near_conf_min);
        const bool near_goal_force = (area >= cfg_.goal_near_force_area_ratio) &&
                                     (abs_off <= cfg_.goal_near_force_offset);
        const bool near_goal = near_goal_base || near_goal_force;
        if (area > cfg_.goal_area_ratio || near_goal) {
            if (pregoal_start_ == 0) pregoal_start_ = t;
            motion_.set_desired(RobotCmd::Stop);
            decision_reason_ = near_goal ? "goal:near_hold" : "goal:close_hold";
            const int hold_gate_ms = near_goal ? cfg_.goal_near_ms : cfg_.pregoal_ms;
            if ((t - pregoal_start_) > hold_gate_ms) {
                if (area > cfg_.goal_area_ratio) {
                    motion_.hold_stop_for(cfg_.goal_hold_ms);
                    goal_latched_ = true;
                    goal_release_lost_ = 0;
                    has_target_ = false;
                    pregoal_start_ = 0;
                    decision_reason_ = "goal:stop_and_latch";
                } else {
                    // Near goal: brief settle pause only, no hard latch.
                    motion_.hold_stop_for(cfg_.stop_hold_ms);
                    pregoal_start_ = 0;
                    decision_reason_ = "goal:near_pause";
                }
            }
            return;
        }
        pregoal_start_ = 0;
        if (turn_state_ == 0) {
            if (abs_off > turn_deadband) {
                turn_state_ = (offset < 0.0) ? -1 : 1;
            }
        } else if (turn_state_ < 0) {
            if (offset > turn_flip) {
                turn_state_ = 1;
            } else if (abs_off < turn_exit) {
                turn_state_ = 0;
            }
        } else {
            if (offset < -turn_flip) {
                turn_state_ = -1;
            } else if (abs_off < turn_exit) {
                turn_state_ = 0;
            }
        }

        if (abs_off > turn_strong) {
            motion_.set_desired(offset < 0 ? RobotCmd::RotL : RobotCmd::RotR);
            decision_reason_ = (offset < 0)
                ? (close_track ? "track:strong_rot_l_close" : "track:strong_rot_l")
                : (close_track ? "track:strong_rot_r_close" : "track:strong_rot_r");
        } else if (turn_state_ < 0) {
            motion_.set_desired(RobotCmd::RotL);
            decision_reason_ = close_track ? "track:rot_l_close" : "track:rot_l";
        } else if (turn_state_ > 0) {
            motion_.set_desired(RobotCmd::RotR);
            decision_reason_ = close_track ? "track:rot_r_close" : "track:rot_r";
        } else {
            if (close_track && lost_ >= cfg_.close_blind_hold_lost_frames) {
                motion_.set_desired(RobotCmd::Stop);
                decision_reason_ = "track:close_blind_hold";
            } else if (target_.source == "fast") {
                const bool vlm_no_ball_recent = (last_vlm_no_ball_ms_ > 0) && ((t - last_vlm_no_ball_ms_) < cfg_.vlm_negative_hold_ms);
                const bool fast_probe = fast_target_active_ &&
                                        fast_target_acquired_ms_ > 0 &&
                                        ((t - fast_target_acquired_ms_) < cfg_.fast_probe_ms) &&
                                        !ai_confirm_recent_;
                bool vlm_match_recent = false;
                if (last_vlm_has_bbox_ && last_vlm_ball_ms_ > 0 && ((t - last_vlm_ball_ms_) < cfg_.vlm_target_match_ms)) {
                    const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);
                    const cv::Rect tb = target_.bbox & frame_rect;
                    const cv::Rect vb = last_vlm_bbox_ & frame_rect;
                    if (tb.width > 1 && tb.height > 1 && vb.width > 1 && vb.height > 1) {
                        vlm_match_recent = rect_iou(tb, vb) >= cfg_.vlm_target_iou_min;
                    }
                }
                const bool ai_ball_fresh = ai_confirm_recent_ && last_vlm_ball_ && last_vlm_rx_ms_ > 0 &&
                                           ((t - last_vlm_rx_ms_) < cfg_.vlm_target_match_ms);
                const bool fast_grace = fast_target_active_ && fast_target_acquired_ms_ > 0 &&
                                        ((t - fast_target_acquired_ms_) < cfg_.fast_forward_grace_ms) &&
                                        target_.conf >= cfg_.fast_grace_conf_min &&
                                        area >= cfg_.fast_grace_min_area &&
                                        abs_off <= cfg_.fast_grace_max_offset &&
                                        lost_ <= 2;
                const bool fast_quality_move = target_.conf >= cfg_.fast_move_conf_min &&
                                               area >= cfg_.fast_move_area_min;
                const bool fast_no_ball_override = target_.conf >= cfg_.fast_no_ball_override_conf &&
                                                   area >= cfg_.fast_no_ball_override_area;
                const bool allow_fast_fwd = (!vlm_no_ball_recent || fast_no_ball_override) &&
                                            (vlm_match_recent || (ai_ball_fresh && fast_quality_move) || fast_grace || fast_no_ball_override);
                if (fast_probe) {
                    motion_.set_desired(RobotCmd::Stop);
                    decision_reason_ = "track:fast_probe";
                } else if (!allow_fast_fwd) {
                    motion_.set_desired(RobotCmd::Stop);
                    decision_reason_ = vlm_no_ball_recent ? "track:wait_vlm_no_ball" : "track:wait_fast_quality";
                } else {
                    motion_.set_desired(RobotCmd::Fwd);
                    decision_reason_ = fast_grace ? "track:fwd_fast_grace" : "track:fwd_fast_vlm";
                }
            } else if (target_.source == "vlm") {
                const bool fast_recent_confirm = last_fast_seen_ms_ > 0 &&
                                                 (t - last_fast_seen_ms_) < cfg_.fast_confirm_max_age_ms &&
                                                 last_fast_conf_ >= cfg_.fast_lock_conf_min;
                if (!fast_recent_confirm) {
                    motion_.set_desired(RobotCmd::Stop);
                    decision_reason_ = "track:wait_fast_confirm";
                } else {
                    motion_.set_desired(RobotCmd::Fwd);
                    decision_reason_ = "track:fwd_vlm_fast_confirmed";
                }
            } else {
                motion_.set_desired(RobotCmd::Fwd);
                decision_reason_ = "track:fwd";
            }
        }
    }

    void stuck_step(const cv::Mat& frame, int64_t t) {
        if (!auto_) {
            prev_gray_.release();
            stuck_count_ = 0;
            stuck_note_ = "manual";
            return;
        }
        if (t < unstuck_until_) {
            motion_.set_desired(unstuck_right_ ? RobotCmd::RotR : RobotCmd::RotL);
            stuck_note_ = unstuck_right_ ? "unstuck_rot_r" : "unstuck_rot_l";
            return;
        }
        cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        if (!prev_gray_.empty()) {
            const double diff = mean_absdiff_gray(gray, prev_gray_);
            if (motion_.sent_cmd() == RobotCmd::Fwd && diff < cfg_.stuck_diff) ++stuck_count_; else stuck_count_ = 0;
            stuck_note_ = "diff=" + fmt_double(diff, 2);
            if (stuck_count_ >= cfg_.stuck_frames && (t - last_unstuck_) > 1800) {
                last_unstuck_ = t;
                stuck_count_ = 0;
                unstuck_right_ = !unstuck_right_;
                unstuck_until_ = t + cfg_.unstuck_rotate_ms;
                motion_.hold_stop_for(cfg_.stop_hold_ms);
                push_event("stuck_detected -> hold+rotate");
            }
        }
        prev_gray_ = gray;
    }

    void draw(cv::Mat& frame, int64_t t) {
        const int64_t vlm_fresh_ms = (last_vlm_rx_ms_ > 0) ? (t - last_vlm_rx_ms_) : -1;
        if (has_target_) {
            const cv::Scalar c = target_.source == "vlm" ? cv::Scalar(40, 220, 40) : cv::Scalar(0, 180, 255);
            cv::rectangle(frame, target_.bbox, c, 2);
        }

        std::vector<std::pair<std::string, cv::Scalar>> lines;
        lines.push_back({std::string("[SPACE] auto: ") + (auto_ ? "ON" : "OFF") + " phase: " + phase_, cv::Scalar(0, 255, 255)});
        lines.push_back({std::string("desired: ") + cmd_name(motion_.desired_cmd()) + " sent: " + cmd_name(motion_.sent_cmd()) + " holdMs: " + std::to_string(motion_.hold_remaining_ms(t)), cv::Scalar(255, 220, 90)});
        lines.push_back({"decision: " + shorten(decision_reason_, 76), cv::Scalar(220, 245, 180)});
        lines.push_back({"target: " + std::string(has_target_ ? "YES" : "NO") + " cand=" + std::to_string(has_candidate_ ? candidate_count_ : 0) +
                         " src=" + (has_target_ ? target_.source : "none") + " conf=" + fmt_double(has_target_ ? target_.conf : 0.0, 2) +
                         " off=" + fmt_double(last_offset_, 3) + " area=" + fmt_double(last_area_ratio_, 3) + " lost=" + std::to_string(lost_), cv::Scalar(180, 240, 240)});
        lines.push_back({"vlm: ball=" + std::string(last_vlm_ball_ ? "1" : "0") + " conf=" + fmt_double(last_vlm_conf_, 2) +
                         " vlmFreshMs=" + std::to_string(vlm_fresh_ms) +
                         " aiSeen=" + std::string(ai_confirm_recent_ ? "1" : "0"), cv::Scalar(200, 180, 255)});
        lines.push_back({"unlock_reason: " + unlock_reason_, cv::Scalar(255, 210, 180)});
        lines.push_back({"fast: conf=" + fmt_double(last_fast_conf_, 2) +
                         " ageMs=" + std::to_string((last_fast_seen_ms_ > 0) ? (t - last_fast_seen_ms_) : -1), cv::Scalar(180, 220, 255)});
        lines.push_back({"stream: frameAgeMs=" + std::to_string(frame_age_ms_), cv::Scalar(180, 210, 255)});
        lines.push_back({"stuck: cnt=" + std::to_string(stuck_count_) + " note=" + shorten(stuck_note_, 42), cv::Scalar(220, 220, 170)});
        lines.push_back({"http: " + std::to_string(motion_.last_status_code()) + " " + shorten(motion_.last_error(), 48), cv::Scalar(220, 220, 130)});
        lines.push_back({"url: " + shorten(motion_.last_url(), 76), cv::Scalar(170, 210, 170)});
        lines.push_back({"key: " + std::to_string(last_key_) + " sharpness: " + std::to_string(static_cast<int>(sharpness_lap_var(frame))), cv::Scalar(210, 210, 210)});
        lines.push_back({"manual: mode=" + std::string(manual_latch_active_ ? "LATCH" : "MOMENTARY") +
                         " W/X=fwd/back A/D/Z/C=rot [M] mode [1|S] stop [Space] auto [Q] quit", cv::Scalar(180, 180, 180)});
        lines.push_back({"search: mode=" + std::string(cfg_.search_single_direction ? "single" : "alternate") +
                         " dir=" + std::string(search_right_ ? "R" : "L"), cv::Scalar(180, 210, 210)});
        lines.push_back({"goal: latched=" + std::string(goal_latched_ ? "1" : "0") +
                         " nearA=" + fmt_double(cfg_.goal_near_area_ratio, 2), cv::Scalar(200, 230, 180)});

        for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
            lines.push_back({"event: " + shorten(*it, 82), cv::Scalar(150, 255, 150)});
        }

        const int panel_w = std::min(frame.cols, 920);
        const int panel_h = std::min(frame.rows, 14 + static_cast<int>(lines.size()) * 18);
        if (panel_w > 10 && panel_h > 10) {
            cv::Mat roi = frame(cv::Rect(0, 0, panel_w, panel_h));
            cv::Mat overlay(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
            cv::addWeighted(overlay, 0.52, roi, 0.48, 0.0, roi);
        }

        const double fs = cfg_.hud_font_scale;
        auto hud = [&](int i, const std::string& text, cv::Scalar c) {
            cv::putText(frame, text, cv::Point(10, 22 + i * 18), cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0, 0, 0), 2);
            cv::putText(frame, text, cv::Point(10, 22 + i * 18), cv::FONT_HERSHEY_SIMPLEX, fs, c, 1);
        };
        for (size_t i = 0; i < lines.size(); ++i) {
            hud(static_cast<int>(i), lines[i].first, lines[i].second);
        }
    }

    const Config& cfg_;
    MotionScheduler& motion_;
    AsyncVLMWorker& worker_;
    bool auto_ = false;
    RobotCmd manual_ = RobotCmd::Stop;
    Detection target_;
    bool has_target_ = false;
    std::optional<Detection> candidate_;
    bool has_candidate_ = false;
    int candidate_count_ = 0;
    int lost_ = 0;
    int turn_state_ = 0;
    bool ema_init_ = false;
    double ema_x_ = 0.5;
    double last_offset_ = 0.0;
    double last_area_ratio_ = 0.0;
    bool search_right_ = true;
    bool search_hint_right_ = true;
    bool search_hint_valid_ = false;
    bool search_phase_initialized_ = false;
    bool search_rotating_ = false;
    int search_step_ = 0;
    int64_t search_phase_until_ms_ = 0;
    int64_t reacquire_hold_until_ms_ = 0;
    int64_t last_submit_ = 0, last_target_ms_ = 0, pregoal_start_ = 0;
    cv::Mat prev_gray_;
    int stuck_count_ = 0;
    int64_t last_unstuck_ = 0, unstuck_until_ = 0;
    bool unstuck_right_ = true;
    int last_key_ = -1;
    std::string phase_ = "INIT";
    std::string decision_reason_ = "startup";
    std::string unlock_reason_ = "none";
    std::string stuck_note_ = "n/a";
    int64_t last_vlm_rx_ms_ = 0;
    bool last_vlm_ball_ = false;
    double last_vlm_conf_ = 0.0;
    int64_t last_vlm_ball_ms_ = 0;
    int64_t last_vlm_no_ball_ms_ = 0;
    cv::Rect last_vlm_bbox_;
    bool last_vlm_has_bbox_ = false;
    int64_t last_fast_seen_ms_ = 0;
    double last_fast_conf_ = 0.0;
    int64_t frame_age_ms_ = -1;
    int64_t ai_confirm_ms_ = 0;
    bool ai_confirm_recent_ = false;
    bool goal_latched_ = false;
    int goal_release_lost_ = 0;
    bool fast_target_active_ = false;
    int64_t fast_target_acquired_ms_ = 0;
    int64_t frame_count_ = 0;
    int64_t last_console_log_ms_ = 0;
    std::vector<std::string> events_;
    int64_t manual_until_ms_ = 0;
    bool manual_latch_active_ = false;
    int64_t last_manual_event_ms_ = 0;
    RobotCmd last_manual_event_cmd_ = RobotCmd::Stop;
};

int main() {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    Config cfg;
    if (cfg.verbose_console) {
        console_log("start robot_navigation");
        console_log("stream=" + cfg.mjpeg_url + " cmd=" + cfg.cmd_url + " ollama=" + cfg.ollama_url);
    }
    MJPEGReader reader(cfg.mjpeg_url);
    if (!reader.open()) { std::cerr << "Failed to open stream: " << cfg.mjpeg_url << "\n"; return 1; }

    RobotCommander commander(cfg.cmd_url, cfg.cmd_min_interval_ms, cfg.cmd_timeout_ms);
    MotionScheduler scheduler(
        commander,
        cfg.scheduler_keepalive_ms,
        cfg.pulse_duration_drive_ms,
        cfg.pulse_pause_drive_ms,
        cfg.pulse_duration_rotate_ms,
        cfg.pulse_pause_rotate_ms
    );
    AsyncVLMWorker worker(VLMClient(cfg.ollama_url, cfg.model, cfg.vlm_timeout_ms), cfg.vlm_interval_ms);
    Navigator nav(cfg, scheduler, worker);

    cv::namedWindow("Robot Navigation", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat frame;
        int64_t frame_age_ms = -1;
        if (!reader.read(frame, &frame_age_ms)) continue;
        nav.set_frame_age_ms(frame_age_ms);
        nav.step(frame);
        cv::Mat display = letterbox_resize(frame, cfg.preview_width, cfg.preview_height);
        cv::imshow("Robot Navigation", display);
        const int key = cv::waitKeyEx(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
        if (key >= 0) nav.on_key(key);
    }

    scheduler.hold_stop_for(250);
    scheduler.tick();
    return 0;
}
