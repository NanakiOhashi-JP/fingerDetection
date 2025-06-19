import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf  # TFLite Interpreter のために必要

# --- 1. 設定とモデルのロード ---

# TFLiteモデルのパス (ダウンロードしたモデルのファイル名と場所に合わせてください)
TFLITE_MODEL_PATH = 'finger_counting_model_quantized.tflite'

# モデルが期待する入力画像サイズ
# Colabでの前処理（IMG_HEIGHT, IMG_WIDTH）に合わせる
IMG_HEIGHT = 96
IMG_WIDTH = 96

# 指の本数のクラス数（0-5本なので6クラス）
NUM_CLASSES = 6

# TFLite Interpreter のロード
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()  # テンソルを割り当てる

    # モデルの入出力詳細を取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # モデルの入力形式を確認（ShapeとType）
    print(f"TFLiteモデルの入力形状: {input_details[0]['shape']}")
    print(f"TFLiteモデルの入力データ型: {input_details[0]['dtype']}")

except Exception as e:
    print(f"エラー: TFLiteモデルのロードまたは初期化に失敗しました。パスを確認してください: {TFLITE_MODEL_PATH}")
    print(e)
    exit()

# MediaPipe Hands の初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


# --- 2. 後続の計算・表示を担当する関数 ---
def calculate_and_display_results(left_fingers: int, right_fingers: int, frame_to_display):
    total_fingers = left_fingers + right_fingers
    difference_fingers = abs(left_fingers - right_fingers)

    cv2.putText(frame_to_display, f"Left Hand: {left_fingers}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_to_display, f"Right Hand: {right_fingers}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_to_display, f"Total: {total_fingers}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_to_display, f"Difference: {difference_fingers}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


# --- 3. メイン処理ループ ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラを開けませんでした。カメラが接続されているか、他のアプリで使用されていないか確認してください。")
    exit()

print("カメラを開きました。リアルタイム認識を開始します。'q' キーを押すと終了します。")

while cap.isOpened():
    ret, frame = cap.read() # ret -> bool
    if not ret:
        print("エラー: フレームを取得できませんでした。ストリームを終了します。")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe Hands用にRGBに変換
    results = hands.process(rgb_frame)

    current_left_fingers = 0
    current_right_fingers = 0

    if results.multi_hand_landmarks:
        detected_hands_info = []

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_label = handedness.classification[0].label

            h, w, _ = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size == 0 or hand_roi.shape[0] == 0 or hand_roi.shape[1] == 0:
                continue

            # --- 4. 指カウントモデルによる推論 ---
            # 1. BGR (OpenCV) -> グレースケール変換
            img_gray_for_model = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

            # 2. リサイズ (モデルの入力サイズに合わせる)
            img_resized_for_model = cv2.resize(img_gray_for_model, (IMG_WIDTH, IMG_HEIGHT))

            # 3. データ型変換と正規化 (0-1にスケーリング)
            img_normalized_for_model = img_resized_for_model.astype(np.float32) / 255.0

            # 4. バッチ次元とチャンネル次元の追加 (モデル入力は (1, height, width, 1) を期待)
            img_input_for_model = np.expand_dims(img_normalized_for_model, axis=-1)  # (H, W, 1)
            img_input_for_model = np.expand_dims(img_input_for_model, axis=0)  # (1, H, W, 1)

            # TFLite Interpreter に入力データをセット
            interpreter.set_tensor(input_details[0]['index'], img_input_for_model)

            # 推論実行
            interpreter.invoke()

            # 出力テンソルから結果を取得
            predictions = interpreter.get_tensor(output_details[0]['index'])

            # 最も確率の高いクラスのインデックス（指の本数）を取得
            predicted_count = np.argmax(predictions[0])

            detected_hands_info.append(((x_min + x_max) / 2, mp_label, predicted_count, hand_landmarks))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # 検出した手の情報をX座標でソートする代わりに、MediaPipeの左右判定を使用
        for x_center, mp_label, count, landmarks in detected_hands_info:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # MediaPipeの判定に基づいて左右の手を識別
            if mp_label == "Left":  # MediaPipeは画像上での左右を返すため、反転した画像では実際と逆になる
                current_right_fingers = count  # カメラ映像が反転しているため、"Left"は実際には右手
            elif mp_label == "Right":
                current_left_fingers = count  # カメラ映像が反転しているため、"Right"は実際には左手

    # --- 5. 識別結果の引き渡し ---
    calculate_and_display_results(
        left_fingers=current_left_fingers,
        right_fingers=current_right_fingers,
        frame_to_display=frame
    )

    # --- 6. 結果の表示と終了処理 ---
    cv2.imshow('Finger Counting and Calculation System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("プログラムを終了しました。")