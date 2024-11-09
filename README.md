# GoPro2Map

GoProで撮影した映像からGPSデータを抽出し、地理院地図上にマッピングするツール

## 概要
- GoProのGPSメタデータを抽出
- フレームを一定間隔で切り出し
- 地理院地図上に軌跡とフレーム画像をマッピング

## Requirements
- Python 3.8以上
- ExifTool
- OpenCV
- その他必要なパッケージ（requirements.txtに記載）

## インストール
```bash
# リポジトリのクローン
git clone git@github.com:yoshiyama/GoPro2Map.git

# 依存パッケージのインストール
pip install -r requirements.txt

# ExifToolのインストール（Ubuntuの場合）
sudo apt-get install exiftool
```

## 使用方法
```bash
python frame2map_gsimap.py video_file.MP4 output_dir [options]
```

### オプション
- `--interval`: フレーム抽出間隔（秒）
- `--radius`: 道路マッチングの検索半径（メートル）
- `--lat`: GPS情報がない場合の緯度指定
- `--lon`: GPS情報がない場合の経度指定
- `--zoom`: 地図の初期ズームレベル（デフォルト: 17）

### 出力
- マップHTML（output/map.html）
- 抽出フレーム（output/frames/）
- GPSデータ（output/gps_data.csv）

## 特徴
- 地理院地図の利用
- 道路ネットワークへのマッチング
- フレームとGPSの同期
- 複数の地図タイプ（標準、写真、淡色）

## ライセンス
このプロジェクトはMITライセンスの下で公開されています。

## 謝辞
- 地理院地図を利用させていただいています
- OpenStreetMapのデータを利用しています

## 開発者
- 山本義幸
- 愛知工業大学工学部社会基盤学科

## コントリビューション
1. Forkする
2. Featureブランチを作成
3. 変更をCommit
4. ブランチをPush
5. Pull Requestを作成

## トラブルシューティング
- GPSデータが取得できない場合
  - `--lat`と`--lon`オプションで位置を手動指定できます
- フレーム抽出に失敗する場合
  - OpenCVのバージョンを確認してください
- 地図表示に問題がある場合
  - ブラウザのコンソールでエラーを確認してください

## 更新履歴
- v1.0.0: 初期リリース
  - 基本的なGPS抽出機能
  - 地理院地図との連携
- v1.1.0: 機能追加
  - 複数地図タイプのサポート
  - GPSデータ手動入力機能
