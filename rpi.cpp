#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    // Load the Haar cascade for face detection
    cv::CascadeClassifier haar_cascade;
    if (!haar_cascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"))) {
        std::cerr << "Error loading Haar cascade." << std::endl;
        return -1;
    }

    // Load the crosshair image (Atman_right.png) and resize it to use as a bounding box overlay
    cv::Mat crosshair = cv::imread("assets/Atman_right.png", cv::IMREAD_UNCHANGED);
    if (crosshair.empty()) {
        std::cerr << "Error loading crosshair image." << std::endl;
        return -1;
    }

    // Load the transparent box image for displaying text
    cv::Mat transparent_box = cv::imread("assets/Transparent_Box.png", cv::IMREAD_UNCHANGED);
    if (transparent_box.empty()) {
        std::cerr << "Error loading transparent box image." << std::endl;
        return -1;
    }

    // Load the text data from Atman.txt
    std::ifstream file("Database/Atman.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening text file." << std::endl;
        return -1;
    }
    std::vector<std::string> text_lines;
    std::string line;
    while (std::getline(file, line)) {
        text_lines.push_back(line);
    }
    file.close();

    // Start capturing video from the webcam (source=0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture." << std::endl;
        return -1;
    }

    // Define the target window size for smartphone viewing
    int target_width = 1920; // Adjust this width for VR headset use
    int target_height = 1080; // Adjust this height for VR headset use

    while (true) {
        // Capture frame-by-frame
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Get frame dimensions
        int frame_height = frame.rows;
        int frame_width = frame.cols;

        // Draw a blue circle at the center of the screen
        int center_x = frame_width / 2;
        int center_y = frame_height / 2;
        cv::circle(frame, cv::Point(center_x, center_y), 10, cv::Scalar(255, 0, 0), -1); // Blue circle

        // Convert frame to grayscale for Haar cascade detection
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

        // Detect faces using Haar cascade
        std::vector<cv::Rect> detections;
        haar_cascade.detectMultiScale(gray_frame, detections, 1.1, 5, 0, cv::Size(30, 30));

        // Flag to check if center circle is inside any detection
        bool center_in_detection = false;

        // Loop through each detection and overlay the crosshair image (bounding box)
        for (const auto& detection : detections) {
            int x = detection.x;
            int y = detection.y;
            int w = detection.width;
            int h = detection.height;

            // Check if the center circle is inside the detection box
            if (x <= center_x && center_x <= (x + w) && y <= center_y && center_y <= (y + h)) {
                center_in_detection = true;
            }

            // Resize the crosshair to match the size of the bounding box
            cv::Mat crosshair_resized;
            cv::resize(crosshair, crosshair_resized, cv::Size(w, h));

            // Ensure the crosshair is not placed outside of the frame boundaries
            if (x < 0 || y < 0 || (x + w) > frame_width || (y + h) > frame_height) {
                continue;
            }

            // Extract region of interest (ROI) where crosshair will be placed
            cv::Mat roi = frame(cv::Rect(x, y, w, h));

            // Split the crosshair into its color and alpha channel
            std::vector<cv::Mat> channels(4);
            cv::split(crosshair_resized, channels);
            cv::Mat crosshair_rgb;
            cv::merge(channels.begin(), channels.begin() + 3, crosshair_rgb);
            cv::Mat crosshair_alpha = channels[3] / 255.0;

            // Blend the crosshair with the ROI based on alpha channel
            for (int i = 0; i < roi.rows; ++i) {
                for (int j = 0; j < roi.cols; ++j) {
                    for (int c = 0; c < 3; ++c) {
                        roi.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(
                            crosshair_alpha.at<uchar>(i, j) * crosshair_rgb.at<cv::Vec3b>(i, j)[c] +
                            (1 - crosshair_alpha.at<uchar>(i, j)) * roi.at<cv::Vec3b>(i, j)[c]
                        );
                    }
                }
            }
        }

        // If the center circle is inside Atman_right.png (detected bounding box), overlay the text
        if (center_in_detection) {
            // Resize the transparent box to accommodate the text
            int box_height = transparent_box.rows;
            int box_width = transparent_box.cols;
            int box_width_resized = 300; // Reduced size of the text box
            double scale = static_cast<double>(box_width_resized) / box_width;
            int box_height_resized = static_cast<int>(box_height * scale);
            cv::Mat transparent_box_resized;
            cv::resize(transparent_box, transparent_box_resized, cv::Size(box_width_resized, box_height_resized));

            // Position to overlay the transparent box
            int box_pos_x = 20;
            int box_pos_y = frame_height - box_height_resized;

            // Overlay the transparent box on the frame
            cv::Mat box_roi = frame(cv::Rect(box_pos_x, box_pos_y, box_width_resized, box_height_resized));

            // Split the transparent box into its color and alpha channel
            std::vector<cv::Mat> box_channels(4);
            cv::split(transparent_box_resized, box_channels);
            cv::Mat box_rgb;
            cv::merge(box_channels.begin(), box_channels.begin() + 3, box_rgb);
            cv::Mat box_alpha = box_channels[3] / 255.0;

            // Blend the transparent box with the ROI based on alpha channel
            for (int i = 0; i < box_roi.rows; ++i) {
                for (int j = 0; j < box_roi.cols; ++j) {
                    for (int c = 0; c < 3; ++c) {
                        box_roi.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(
                            box_alpha.at<uchar>(i, j) * box_rgb.at<cv::Vec3b>(i, j)[c] +
                            (1 - box_alpha.at<uchar>(i, j)) * box_roi.at<cv::Vec3b>(i, j)[c]
                        );
                    }
                }
            }

            // Overlay the text on the transparent box
            int y_offset = box_pos_y + 50;
            for (const auto& text_line : text_lines) {
                cv::putText(frame, text_line, cv::Point(box_pos_x + 15, y_offset), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                y_offset += 20;
            }
        }

        // Create a side-by-side view for VR
        cv::Mat sbs_frame;
        cv::hconcat(frame, frame, sbs_frame);

        // Resize the side-by-side frame to the target window size
        cv::Mat resized_frame;
        cv::resize(sbs_frame, resized_frame, cv::Size(target_width, target_height));

        // Display the frame with crosshair overlays and text if applicable in a resized window
        cv::imshow("Crosshair Overlay", resized_frame);

        // Exit on 'q' key press
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
