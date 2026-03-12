#include <iostream>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif

    cv::VideoCapture cap;

    const std::vector<std::pair<int, std::string>> backends = {
        {cv::CAP_DSHOW, "DSHOW"},
        {cv::CAP_MSMF, "MSMF"},
        {cv::CAP_ANY, "ANY"},
    };

    std::string used_backend;
    int used_index = -1;
    const int max_camera_index = 5;

    for (int camera_index = 0; camera_index <= max_camera_index && !cap.isOpened(); ++camera_index) {
        for (const auto& [backend, name] : backends) {
            cap.release();
            const bool opened = (backend == cv::CAP_ANY) ? cap.open(camera_index) : cap.open(camera_index, backend);
            if (opened) {
                used_index = camera_index;
                used_backend = name;
                break;
            }
        }
    }

    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть камеру (индексы 0.." << max_camera_index << ") ни через DSHOW, ни через MSMF.\n";
        std::cerr << "Проверь, что камера подключена и не занята другим приложением.\n";
        return 1;
    }

    std::cout << "Камера открыта: index=" << used_index << ", backend=" << used_backend << "\n";

    int cuda_device_count = 0;
    try {
        cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
        std::cout << "CUDA devices (OpenCV): " << cuda_device_count << "\n";
    } catch (const cv::Exception& e) {
        cuda_device_count = 0;
        std::cerr << "Не удалось проверить CUDA через OpenCV: " << e.what() << "\n";
        std::cerr << "DNN будет запущен на CPU.\n";
    }

    const std::vector<std::string> class_names = {
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };

    const auto proto_path = (std::filesystem::path("models") / "MobileNetSSD_deploy.prototxt").string();
    const auto weights_path = (std::filesystem::path("models") / "MobileNetSSD_deploy.caffemodel").string();

    cv::dnn::DetectionModel detector;
    bool detector_ready = false;
    bool detector_enabled = false;
    bool detector_using_cuda = false;
    bool detector_runtime_error_reported = false;

    if (std::filesystem::exists(proto_path) && std::filesystem::exists(weights_path)) {
        try {
            cv::dnn::Net net = cv::dnn::readNetFromCaffe(proto_path, weights_path);
            detector = cv::dnn::DetectionModel(net);
            detector.setInputParams(0.007843f, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);

            if (cuda_device_count > 0) {
                try {
                    detector.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    detector.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    detector_using_cuda = true;
                    std::cout << "DNN backend: CUDA\n";
                } catch (const cv::Exception& e) {
                    std::cerr << "CUDA backend недоступен, fallback на CPU: " << e.what() << "\n";
                    detector.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    detector_using_cuda = false;
                    std::cout << "DNN backend: CPU\n";
                }
            } else {
                detector.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                detector_using_cuda = false;
                std::cout << "CUDA недоступна. DNN backend: CPU\n";
            }

            detector_ready = true;
            detector_enabled = true;
            std::cout << "Распознавание включено (MobileNet-SSD). Клавиша D: вкл/выкл.\n";
        } catch (const cv::Exception& e) {
            std::cerr << "Не удалось загрузить модель распознавания: " << e.what() << "\n";
            std::cerr << "Приложение продолжит работу без распознавания.\n";
        }
    } else {
        std::cout << "Модель не найдена. Добавь файлы:\n";
        std::cout << "  " << proto_path << "\n";
        std::cout << "  " << weights_path << "\n";
        std::cout << "После этого распознавание включится автоматически.\n";
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Пустой кадр\n";
            break;
        }

        cv::flip(frame, frame, 1);

        if (detector_enabled) {
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
            bool detect_ok = false;

            try {
                detector.detect(frame, class_ids, confidences, boxes, 0.5f, 0.4f);
                detect_ok = true;
            } catch (const cv::Exception& e) {
                if (detector_using_cuda) {
                    std::cerr << "Ошибка DNN на CUDA, переключаюсь на CPU: " << e.what() << "\n";
                    try {
                        detector.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                        detector.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                        detector_using_cuda = false;
                        detector.detect(frame, class_ids, confidences, boxes, 0.5f, 0.4f);
                        detect_ok = true;
                        std::cout << "DNN backend: CPU (runtime fallback)\n";
                    } catch (const cv::Exception& cpu_e) {
                        if (!detector_runtime_error_reported) {
                            std::cerr << "Ошибка DNN и на CPU: " << cpu_e.what() << "\n";
                            detector_runtime_error_reported = true;
                        }
                    }
                } else if (!detector_runtime_error_reported) {
                    std::cerr << "Ошибка DNN на CPU: " << e.what() << "\n";
                    detector_runtime_error_reported = true;
                }
            }

            if (detect_ok) {
                for (size_t i = 0; i < class_ids.size(); ++i) {
                    const int class_id = class_ids[i];
                    const int score = static_cast<int>(confidences[i] * 100.0f);

                    std::string class_name = "id=" + std::to_string(class_id);
                    if (class_id >= 0 && class_id < static_cast<int>(class_names.size())) {
                        class_name = class_names[class_id];
                    }

                    const std::string label = class_name + " " + std::to_string(score) + "%";

                    cv::rectangle(frame, boxes[i], cv::Scalar(40, 220, 40), 2);
                    int baseline = 0;
                    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
                    const int text_x = std::max(boxes[i].x, 0);
                    const int text_y = std::max(boxes[i].y - 8, text_size.height + 6);
                    cv::rectangle(
                        frame,
                        cv::Rect(text_x, text_y - text_size.height - 6, text_size.width + 8, text_size.height + 8),
                        cv::Scalar(40, 220, 40),
                        cv::FILLED
                    );
                    cv::putText(
                        frame,
                        label,
                        cv::Point(text_x + 4, text_y - 4),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.55,
                        cv::Scalar(0, 0, 0),
                        1
                    );
                }
            }
        }

        std::string detector_status;
        if (detector_enabled) {
            detector_status = detector_using_cuda ? "[D] Detection ON (CUDA)" : "[D] Detection ON (CPU)";
        } else {
            detector_status = "[D] Detection OFF";
        }
        cv::putText(frame, detector_status, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow("USB Camera", frame);

        const int key = cv::waitKey(1);
        if ((key == 'd' || key == 'D') && detector_ready) {
            detector_enabled = !detector_enabled;
        }
        if (key == 'q' || key == 'Q' || key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
