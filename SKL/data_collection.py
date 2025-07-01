import cv2
import os
import glob
from collections import defaultdict

def record_and_split_frames():
    """
    Webカメラで動画を撮影し、終了後にフレーム単位で画像に分割する関数。
    """
    # --- 1. 動画撮影パート ---

    # プレフィックスごとの動画カウンター (例: '1'が入力された回数)
    video_counts = defaultdict(int)
    # 撮影した動画ファイル名を保存するリスト
    recorded_files = []

    # Webカメラを起動 (0はデフォルトのカメラ)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ エラー: Webカメラを起動できませんでした。")
        return

    print("📷 Webカメラを起動しました。")

    while True:
        # 待機中の画面を表示
        ret, frame = cap.read()
        if not ret:
            print("❌ エラー: フレームを取得できませんでした。")
            break
        
        # 画面に操作説明を表示
        cv2.putText(frame, "Press 0-5 to Record, or 'q' to Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Webcam - Press key to start", frame)

        # キー入力を待機
        key = cv2.waitKey(0) & 0xFF
        char_key = chr(key)

        # '0'～'5' が入力されたら撮影開始
        if '0' <= char_key <= '5':
            prefix = char_key
            video_counts[prefix] += 1
            filename = f"{prefix}-{video_counts[prefix]}.avi"
            recorded_files.append(filename)
            
            # 動画保存の設定
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20.0  # フレームレート
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

            cv2.destroyWindow("Webcam - Press key to start") # 待機ウィンドウを閉じる
            print(f"\n▶️ 撮影を開始します... (ファイル名: {filename})")
            print("🔴 撮影ウィンドウで 'q' を押すと撮影を終了します。")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ エラー: フレームのキャプチャに失敗しました。")
                    break
                
                # 映像をファイルに書き込む
                out.write(frame)
                
                # 撮影中の映像を表示
                cv2.imshow(f"Rec: {filename} (Press 'q' to stop)", frame)

                # 'q'が押されたら撮影終了
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # リソースを解放
            out.release()
            cv2.destroyAllWindows()
            print(f"⏹️ 撮影を終了し、'{filename}' を保存しました。")
            print("\n----------------------------------------")
            print("次の操作を入力してください...")


        # 'q' が入力されたらプログラム終了
        elif char_key == 'q':
            print("\nプログラムを終了します。")
            break

    # Webカメラを解放
    cap.release()
    cv2.destroyAllWindows()

    # --- 2. フレーム分割パート ---

    if not recorded_files:
        print("\nフレーム分割対象の動画がありませんでした。")
        return

    print("\n🎞️ 撮影した動画をフレームに分割します...")
    print("----------------------------------------")

    for video_path in recorded_files:
        # フォルダ名を設定 (例: 1-1.avi -> 1-1)
        folder_name = os.path.splitext(video_path)[0]
        
        # フォルダを作成 (存在しない場合)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"📁 '{folder_name}' フォルダを作成しました。")

        # 動画を読み込み
        vid_cap = cv2.VideoCapture(video_path)
        if not vid_cap.isOpened():
            print(f"❌ エラー: '{video_path}' を開けませんでした。")
            continue

        frame_count = 0
        while True:
            ret, frame = vid_cap.read()
            if not ret:
                break  # 動画の最後でループを抜ける
            
            # フレームを画像として保存 (例: 1-1/0.jpg, 1-1/1.jpg ...)
            frame_filename = os.path.join(folder_name, f"{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        
        vid_cap.release()
        print(f"✅ '{video_path}' を {frame_count} フレームに分割しました。")

# --- メイン処理の実行 ---
if __name__ == '__main__':
    record_and_split_frames()
    print("\n✨ すべての処理が完了しました。")