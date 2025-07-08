# リアルタイムで指の本数を判別

import cv2
import numpy as np
import pickle
import time

class RealTimeFingerDetector:
    def __init__(self, model_path):
        """保存されたモデルを読み込み"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 予測の安定化用
        self.prediction_history = []
        self.history_size = 5  # 過去5回の予測を保持
        self.last_prediction_time = 0
        self.prediction_interval = 0.5  # 秒間隔で予測
        
        print("リアルタイム指検出器を初期化")
    
    def preprocess_image(self, image, target_size=(64, 64)):
        """画像の前処理"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        return resized.flatten()
    
    def get_stable_prediction(self, current_prediction):
        """予測結果を安定化（ノイズ除去）"""
        self.prediction_history.append(current_prediction)
        
        # 履歴サイズを制限
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # 最頻値を返す（最も多く予測された値）
        if len(self.prediction_history) >= 3:
            return max(set(self.prediction_history), key=self.prediction_history.count)
        else:
            return current_prediction
    
    def run_realtime_detection(self):
        """リアルタイムで指の本数を判別"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("エラー: Webカメラが開けません")
            return
        
        print("リアルタイム指検出を開始")
        print("'q'キーで終了")
        
        current_prediction = 0
        stable_prediction = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 現在時刻
            current_time = time.time()
            
            # 一定間隔で予測実行
            if current_time - self.last_prediction_time > self.prediction_interval:
                try:
                    # 予測実行
                    current_prediction = self.model.predict([self.preprocess_image(frame)])[0]
                    stable_prediction = self.get_stable_prediction(current_prediction)
                    self.last_prediction_time = current_time
                    
                except Exception as e:
                    print(f"予測エラー: {e}")
            
            # 画面に情報を表示
            display_frame = frame.copy()
            
            # 背景を少し暗くして文字を見やすく
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            # 予測結果を表示
            cv2.putText(display_frame, f"Fingers: {stable_prediction}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # 現在の予測値も表示（参考用）
            cv2.putText(display_frame, f"Current: {current_prediction}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 終了方法を表示
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # フレームを表示
            cv2.imshow('Real-time Finger Detection', display_frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("リアルタイム検出を終了しました")

def main():
    model_path = "finger_model_combined_2.pkl"
    
    try:
        detector = RealTimeFingerDetector(model_path)
        detector.run_realtime_detection()
        
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{model_path}' が見つかりません")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()