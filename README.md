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

3. 使用方法：
```markdown
## 使用方法
```bash
python frame2map_gsimap.py video_file.MP4 output_dir [options]

4. 機能や特徴：
```markdown
## 特徴
- 地理院地図の利用
- 道路ネットワークへのマッチング
- フレームとGPSの同期
- 複数の地図タイプ（標準、写真、淡色）

## ライセンス
このプロジェクトは[ライセンス名]の下で公開されています。

## 謝辞
- 地理院地図を利用させていただいています
- [その他の謝辞...]

## 開発者
- [あなたの名前]
- [所属など]

## コントリビューション
1. Forkする
2. Featureブランチを作成
3. 変更をCommit
4. ブランチをPush
5. Pull Requestを作成

## トラブルシューティング
よくある問題と解決方法...

## 更新履歴
- v1.0.0: 初期リリース
- v1.1.0: 機能追加...
