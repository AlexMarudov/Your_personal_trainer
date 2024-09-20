from flask import Flask, request, render_template, send_file, url_for, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

model = YOLO("yolov8m-pose.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, save=False)
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        keypoints_list.append(keypoints)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    cap.release()
    return keypoints_list, timestamps

def align_keypoints(ref_keypoints, user_keypoints, reference_points_idx):
    ref_reference_points = ref_keypoints[reference_points_idx]
    user_reference_points = user_keypoints[reference_points_idx]

    if np.any(np.isnan(ref_reference_points)) or np.any(np.isnan(user_reference_points)):
        return ref_keypoints

    ref_mean = np.mean(ref_reference_points, axis=0)
    user_mean = np.mean(user_reference_points, axis=0)

    aligned_ref_keypoints = ref_keypoints - ref_mean + user_mean
    return aligned_ref_keypoints

def synchronize_videos_with_dtw(ref_keypoints, user_keypoints):
    ref_keypoints_flat = [keypoint.flatten() for keypoint in ref_keypoints]
    user_keypoints_flat = [keypoint.flatten() for keypoint in user_keypoints]

    distance, path = fastdtw(ref_keypoints_flat, user_keypoints_flat, dist=euclidean)

    synchronized_ref_keypoints = [ref_keypoints[i] for i, _ in path]
    synchronized_user_keypoints = [user_keypoints[j] for _, j in path]

    return synchronized_ref_keypoints, synchronized_user_keypoints

def detect_key_moments(keypoints_list, threshold=50):
    key_moments = []
    for i in range(1, len(keypoints_list)):
        prev_keypoints = keypoints_list[i-1]
        curr_keypoints = keypoints_list[i]
        distance = np.linalg.norm(curr_keypoints - prev_keypoints)
        if distance > threshold:
            key_moments.append(i)
    return key_moments

def compare_keypoints(ref_keypoints, user_keypoints):
    distances = []
    for ref, user in zip(ref_keypoints, user_keypoints):
        distance = cosine(ref.flatten(), user.flatten())
        distances.append(distance)

    average_distance = np.mean(distances)

    if average_distance < 0.05:
        return "Excellent!"
    elif average_distance < 0.1:
        return "Okay, but you can do better!"
    else:
        return "Try again!"

def draw_keypoints(frame, user_keypoints, ref_keypoints):
    for keypoint in user_keypoints[5:]:
        x, y = keypoint[:2]
        if x != 0 and y != 0:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    for keypoint in ref_keypoints[5:]:
        x, y = keypoint[:2]
        if x != 0 and y != 0:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

    return frame

def draw_skeleton(frame, keypoints, color):
    connections = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (11, 12), (12, 6), (6, 5),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]
    for connection in connections:
        pt1 = keypoints[connection[0]]
        pt2 = keypoints[connection[1]]
        if pt1[0] != 0 and pt1[1] != 0 and pt2[0] != 0 and pt2[1] != 0:
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)

def draw_evaluation(frame, evaluation):
    cv2.putText(frame, evaluation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration

def crop_video(video_path, start_time, end_time, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_index <= end_frame:
            out.write(frame)

        frame_index += 1

    cap.release()
    out.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='video/mp4')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'ref_file' not in request.files or 'user_file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    ref_file = request.files['ref_file']
    user_file = request.files['user_file']

    if ref_file.filename == '' or user_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if ref_file and allowed_file(ref_file.filename) and user_file and allowed_file(user_file.filename):
        ref_filename = secure_filename(ref_file.filename)
        user_filename = secure_filename(user_file.filename)
        ref_file_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
        user_file_path = os.path.join(app.config['UPLOAD_FOLDER'], user_filename)
        ref_file.save(ref_file_path)
        user_file.save(user_file_path)

        ref_keypoints, ref_timestamps = extract_keypoints(ref_file_path)
        user_keypoints, user_timestamps = extract_keypoints(user_file_path)

        ref_key_moments = detect_key_moments(ref_keypoints)
        user_key_moments = detect_key_moments(user_keypoints)

        ref_duration = get_video_duration(ref_file_path)

        center_key_moment = user_key_moments[len(user_key_moments) // 2]
        center_time = user_timestamps[center_key_moment] / 1000

        start_time = center_time - ref_duration / 2
        end_time = center_time + ref_duration / 2
        cropped_user_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_user.mp4')
        crop_video(user_file_path, start_time, end_time, cropped_user_file_path)

        cropped_user_keypoints, cropped_user_timestamps = extract_keypoints(cropped_user_file_path)

        synchronized_ref_keypoints, synchronized_user_keypoints = synchronize_videos_with_dtw(
            ref_keypoints, cropped_user_keypoints
        )

        reference_points_idx = [9, 10, 11, 12]
        aligned_ref_keypoints = [align_keypoints(ref, user, reference_points_idx) for ref, user in zip(synchronized_ref_keypoints, synchronized_user_keypoints)]

        evaluation = compare_keypoints(aligned_ref_keypoints, synchronized_user_keypoints)

        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
        cap = cv2.VideoCapture(cropped_user_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #fourcc = cv2.VideoWriter_fourcc(*'h264')
        out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index < len(synchronized_user_keypoints) and frame_index < len(aligned_ref_keypoints):
                frame = draw_keypoints(frame, synchronized_user_keypoints[frame_index], aligned_ref_keypoints[frame_index])
                draw_skeleton(frame, synchronized_user_keypoints[frame_index], (0, 255, 0))
                draw_skeleton(frame, aligned_ref_keypoints[frame_index], (0, 0, 255))
                frame = draw_evaluation(frame, evaluation)
                out.write(frame)

            frame_index += 1

        cap.release()
        out.release()

        # Проверка существования файла перед отправкой
        if os.path.exists(output_file_path):
            video_url = url_for('static', filename='uploads/output.mp4')
            logging.debug(f"Video URL: {video_url}")
            return jsonify({"evaluation": evaluation, "video_url": video_url})
        else:
            return jsonify({"error": "Output file not found"}), 500

    return jsonify({"error": "Invalid file"}), 400

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
