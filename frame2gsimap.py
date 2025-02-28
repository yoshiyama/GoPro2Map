import cv2
import pandas as pd
import numpy as np
import osmnx as ox
import folium
from folium import plugins
import subprocess
import json
import os
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from shapely.geometry import Point, LineString
from PIL import Image
import mercantile


class GoproMetadataExtractor:
    """
    A class to extract GPS metadata from GoPro video files.

    Attributes:
        temp_dir (str): Directory for temporary metadata and debug files
        debug_mode (bool): Enable/disable debug output and validation
    """

    def __init__(self,
                 temp_dir="temp_metadata",
                 debug_mode=False):
        """
        Initialize the GoPro metadata extractor.

        Args:
            temp_dir (str): Directory for temporary files and debug output
            debug_mode (bool): Enable detailed debug output and validation
        """
        self.temp_dir = temp_dir
        self.debug_mode = debug_mode

        # Create necessary directories
        self.debug_dir = os.path.join(temp_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

    def extract_metadata(self, video_path):
        """Extract GPS metadata from GoPro video with unified timestamp handling"""
        try:
            print("Extracting GPS metadata from video...")

            # Get video creation time
            cmd_basic = ['exiftool', '-CreateDate', '-s', '-s', '-s', video_path]
            result_basic = subprocess.run(cmd_basic, capture_output=True, text=True)
            create_date_str = result_basic.stdout.strip()

            try:
                video_start_time = datetime.strptime(create_date_str, '%Y:%m:%d %H:%M:%S')
                video_start_time = video_start_time.replace(tzinfo=timezone.utc)
            except ValueError:
                print(f"Warning: Could not parse create date: {create_date_str}")
                video_start_time = datetime.now(timezone.utc)

            # Extract GPS data
            cmd = ['exiftool', '-ee', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout

            if self.debug_mode:
                debug_path = os.path.join(self.debug_dir, "raw_metadata.txt")
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(output)

            print("Parsing GPS metadata...")
            gps_points = []
            current_point = {}

            for line in output.split('\n'):
                if ':' not in line:
                    continue

                key, value = [x.strip() for x in line.split(':', 1)]

                try:
                    if 'Time Stamp' in key:
                        try:
                            # Store seconds as float and convert to datetime later
                            seconds = float(value)
                            current_point['timestamp'] = seconds
                        except ValueError:
                            continue
                    elif 'GPS Latitude' in key and 'deg' in value:
                        parts = value.replace('deg', '').replace("'", '').replace('"', '').split()
                        degrees = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2])
                        direction = parts[3]

                        latitude = degrees + minutes / 60 + seconds / 3600
                        if direction == 'S':
                            latitude = -latitude
                        current_point['latitude'] = latitude

                    elif 'GPS Longitude' in key and 'deg' in value:
                        parts = value.replace('deg', '').replace("'", '').replace('"', '').split()
                        degrees = float(parts[0])
                        minutes = float(parts[1])
                        seconds = float(parts[2])
                        direction = parts[3]

                        longitude = degrees + minutes / 60 + seconds / 3600
                        if direction == 'W':
                            longitude = -longitude
                        current_point['longitude'] = longitude

                    elif 'GPS Altitude' in key:
                        alt_value = value.split()[0]
                        current_point['altitude'] = float(alt_value)
                    elif 'GPS Speed' in key and '3D' not in key:
                        current_point['speed'] = float(value)

                    # Save point when we have all required data
                    if all(k in current_point for k in ['timestamp', 'latitude', 'longitude', 'altitude', 'speed']):
                        gps_points.append(current_point.copy())
                        current_point = {}

                except (ValueError, IndexError) as e:
                    if self.debug_mode:
                        print(f"Warning: Could not parse line: {line} - {str(e)}")
                    continue

            if not gps_points:
                print("No GPS points found in the video, trying backup method...")
                return self.extract_metadata_backup(video_path)

            # Create DataFrame and sort by timestamp (still in seconds)
            df = pd.DataFrame(gps_points)
            df = df.sort_values('timestamp')

            # Now convert timestamps to datetime
            min_timestamp = df['timestamp'].min()
            df['timestamp'] = df['timestamp'].apply(
                lambda x: video_start_time + timedelta(seconds=(x - min_timestamp))
            )

            # Remove duplicate points and interpolate missing values
            df = df.drop_duplicates(subset=['timestamp'])
            df = df.interpolate(method='linear')

            if self.debug_mode:
                debug_df_path = os.path.join(self.debug_dir, "processed_gps_data.csv")
                df.to_csv(debug_df_path, index=False)

                report = {
                    "total_points": len(df),
                    "time_range": {
                        "start": str(df['timestamp'].min()),
                        "end": str(df['timestamp'].max()),
                        "duration_seconds": (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
                    },
                    "coordinate_bounds": {
                        "latitude": {"min": df['latitude'].min(), "max": df['latitude'].max()},
                        "longitude": {"min": df['longitude'].min(), "max": df['longitude'].max()},
                        "altitude": {"min": df['altitude'].min(), "max": df['altitude'].max()}
                    },
                    "speed_stats": {
                        "min": df['speed'].min(),
                        "max": df['speed'].max(),
                        "mean": df['speed'].mean()
                    }
                }

                report_path = os.path.join(self.debug_dir, "gps_validation_report.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)

            print(f"\nGPS Track Summary:")
            print(f"Total points: {len(df)}")
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Latitude range: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
            print(f"Longitude range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
            print(f"Altitude range: {df['altitude'].min():.1f}m to {df['altitude'].max():.1f}m")
            print(f"Speed range: {df['speed'].min():.1f}m/s to {df['speed'].max():.1f}m/s")

            return df

        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.extract_metadata_backup(video_path)

    def extract_metadata_backup(self, video_path):
        """
        Fallback method for metadata extraction using MoviePy.

        Args:
            video_path (str): Path to the video file

        Returns:
            pandas.DataFrame or None: GPS data if available, None otherwise
        """
        from moviepy.editor import VideoFileClip

        try:
            print("Extracting basic metadata using MoviePy...")
            clip = VideoFileClip(video_path)
            duration = clip.duration

            # Since we can't get GPS data through MoviePy
            print("Error: No GPS data found in the video file")
            print("Please specify the location using --lat and --lon options")
            clip.close()
            return None

        except Exception as e:
            print(f"Error extracting metadata with MoviePy: {str(e)}")
            return None

class GoproMapMatcher:
    def __init__(self):
        self.frame_interval = 5  # フレーム抽出間隔（秒）
        self.search_radius = 50  # 道路検索半径（メートル）
        self.map_zoom = 17
        self.metadata_extractor = GoproMetadataExtractor()
        self.start_time = 0  # 処理開始時間（秒）
        self.duration = None  # 処理時間（秒）

    def extract_frames_and_gps(self, video_path, output_dir, batch_size=1000):
        """動画からフレームとGPSデータを抽出（時間同期改善版）"""
        frames_dir = os.path.join(output_dir, 'frames')
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        # ビデオ情報の取得
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # GPSデータの取得
        gps_df = self.metadata_extractor.extract_metadata(video_path)
        if gps_df is None:
            raise Exception("No GPS metadata found in the video")

        # フレーム間隔の計算と調整
        frame_interval = max(1, int(fps * self.frame_interval))
        frame_data = []

        # ビデオとGPSの開始時刻を同期
        video_start_time = gps_df['timestamp'].min()
        video_duration = total_frames / fps

        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Video duration: {video_duration:.1f} seconds")
        print(f"Frame interval: {frame_interval} frames ({self.frame_interval:.1f} seconds)")

        # フレームインデックスの配列を生成
        frame_indices = range(0, total_frames, frame_interval)

        for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue

            # フレームの正確な時刻を計算
            frame_time = frame_idx / fps
            frame_timestamp = video_start_time + timedelta(seconds=frame_time)

            # 最も近いGPSポイントを見つける
            time_diffs = abs(gps_df['timestamp'] - frame_timestamp)
            closest_idx = time_diffs.idxmin()
            time_diff = time_diffs[closest_idx]

            # 大きな時間差がある場合は警告
            if time_diff.total_seconds() > 1.0:  # 1秒以上のずれがある場合
                print(f"\nWarning: Large time difference ({time_diff.total_seconds():.1f}s) at frame {frame_idx}")
                print(f"Frame time: {frame_timestamp}")
                print(f"Closest GPS time: {gps_df['timestamp'][closest_idx]}")

            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)

            try:
                cv2.imwrite(frame_path, frame)

                frame_data.append({
                    'timestamp': gps_df['timestamp'][closest_idx],
                    'frame_path': frame_path,
                    'latitude': gps_df['latitude'][closest_idx],
                    'longitude': gps_df['longitude'][closest_idx],
                    'altitude': gps_df['altitude'][closest_idx],
                    'speed': gps_df['speed'][closest_idx],
                    'frame_time': frame_time,
                    'gps_time': (gps_df['timestamp'][closest_idx] - video_start_time).total_seconds(),
                    'time_diff': time_diff.total_seconds(),
                    'frame_index': frame_idx
                })

            except Exception as e:
                print(f"\nWarning: Error processing frame {frame_idx}: {str(e)}")

        cap.release()

        # 結果をDataFrameに変換
        result_df = pd.DataFrame(frame_data)

        print(f"\nFrame extraction summary:")
        print(f"Total frames extracted: {len(result_df)}")
        print(f"Average time difference: {result_df['time_diff'].mean():.3f} seconds")
        print(f"Max time difference: {result_df['time_diff'].max():.3f} seconds")

        return result_df

    def calculate_distance(self, df):
        """2点間の距離を計算"""
        from math import radians, sin, cos, sqrt, atan2

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth's radius in meters

            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        total_distance = 0
        for i in range(len(df) - 1):
            total_distance += haversine(
                df['latitude'].iloc[i], df['longitude'].iloc[i],
                df['latitude'].iloc[i + 1], df['longitude'].iloc[i + 1]
            )
        return total_distance

    def get_road_network(self, df):
            """地域の道路ネットワークを取得（改善版）"""
            print("Downloading road network...")

            # バウンディングボックスの計算
            center_lat = df['latitude'].mean()
            center_lon = df['longitude'].mean()

            # 範囲を広げる（約1km四方）
            delta_lat = 0.005  # 約500m
            delta_lon = 0.005 / np.cos(np.radians(center_lat))  # 緯度に応じて経度の差を調整

            bbox = (
                center_lon - delta_lon,  # west
                center_lat - delta_lat,  # south
                center_lon + delta_lon,  # east
                center_lat + delta_lat  # north
            )

            try:
                # 最初に小さな範囲で試行
                print(f"Attempting to download road network around {center_lat:.4f}, {center_lon:.4f}")
                G = ox.graph_from_bbox(
                    bbox[3], bbox[1], bbox[2], bbox[0],
                    network_type='drive',
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                    clean_periphery=True
                )
            except Exception as e:
                print(f"First attempt failed, trying with larger area...")
                # 範囲を2倍に広げて再試行
                bbox = (
                    center_lon - delta_lon * 2,
                    center_lat - delta_lat * 2,
                    center_lon + delta_lon * 2,
                    center_lat + delta_lat * 2
                )
                try:
                    G = ox.graph_from_bbox(
                        bbox[3], bbox[1], bbox[2], bbox[0],
                        network_type='drive',
                        simplify=True,
                        retain_all=False,
                        truncate_by_edge=True,
                        clean_periphery=True
                    )
                except Exception as e2:
                    print(f"Error downloading road network: {str(e2)}")
                    print("Trying alternative download method...")
                    # 場所名から取得を試みる
                    try:
                        location = ox.geocode_to_gdf(f"{center_lat}, {center_lon}")
                        G = ox.graph_from_point(
                            (center_lat, center_lon),
                            dist=1000,  # 1km radius
                            network_type='drive',
                            simplify=True
                        )
                    except Exception as e3:
                        raise Exception(f"Failed to download road network: {str(e3)}")

            # グラフをGeoDataFrameに変換
            nodes, edges = ox.graph_to_gdfs(G)
            print(f"Successfully downloaded road network with {len(edges)} road segments")
            return edges

    def match_to_roads(self, matched_df, road_edges):
        """GPS座標を最寄りの道路上に移動（改善版）"""
        print("Matching points to roads...")

        if road_edges.empty:
            print("Warning: No road data available, using original coordinates")
            return matched_df

        road_lines = []
        for idx, row in road_edges.iterrows():
            if isinstance(row.geometry, (LineString, list)):
                road_lines.append(row.geometry)

        if not road_lines:
            print("Warning: No valid road geometries found, using original coordinates")
            return matched_df

        matched_points = []

        for _, point in tqdm(matched_df.iterrows(), desc="Matching GPS points"):
            pt = Point(point['longitude'], point['latitude'])
            min_dist = float('inf')
            best_point = None

            for line in road_lines:
                try:
                    proj_point = line.interpolate(line.project(pt))
                    dist = pt.distance(proj_point)

                    if dist < min_dist:
                        min_dist = dist
                        best_point = proj_point
                except Exception as e:
                    continue

            if best_point and min_dist < self.search_radius * 0.00001:
                matched_points.append({
                    'timestamp': point['timestamp'],
                    'frame_path': point['frame_path'],
                    'latitude': best_point.y,
                    'longitude': best_point.x,
                    'original_lat': point['latitude'],
                    'original_lon': point['longitude'],
                    'altitude': point.get('altitude'),
                    'speed': point.get('speed')
                })
            else:
                # マッチする道路が見つからない場合は元の座標を使用
                matched_points.append({
                    'timestamp': point['timestamp'],
                    'frame_path': point['frame_path'],
                    'latitude': point['latitude'],
                    'longitude': point['longitude'],
                    'original_lat': point['latitude'],
                    'original_lon': point['longitude'],
                    'altitude': point.get('altitude'),
                    'speed': point.get('speed')
                })

        result_df = pd.DataFrame(matched_points)
        print(f"Matched {len(matched_points)} points to roads")
        return result_df

    def create_map(self, matched_df, output_path):
        """マップを作成して画像を配置（地理院地図使用版）"""
        print("Creating map...")

        center_lat = matched_df['latitude'].mean()
        center_lon = matched_df['longitude'].mean()

        # 出力ディレクトリのパスを取得
        output_dir = os.path.dirname(os.path.abspath(output_path))

        # 基本的なHTML構造を作成
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>GPS Track Map</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <style>
                #map {{
                    height: 800px;
                    width: 100%;
                }}
                .custom-popup-image {{
                    max-width: 150px;
                    cursor: pointer;
                    transition: transform 0.3s ease;
                }}
                .custom-popup-image:hover {{
                    transform: scale(1.05);
                }}
                .modal {{
                    display: none;
                    position: fixed;
                    z-index: 1000;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    background-color: rgba(0,0,0,0.9);
                }}
                .modal-content {{
                    margin: auto;
                    display: block;
                    max-width: 90%;
                    max-height: 90vh;
                    margin-top: 2%;
                }}
                .close {{
                    color: #fff;
                    position: absolute;
                    right: 35px;
                    top: 15px;
                    font-size: 40px;
                    font-weight: bold;
                    cursor: pointer;
                }}
                .debug-info {{
                    display: none;
                    color: red;
                    font-size: 10px;
                }}
                .map-type-control {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    z-index: 1000;
                    background: white;
                    padding: 10px;
                    border-radius: 4px;
                    box-shadow: 0 1px 5px rgba(0,0,0,0.4);
                }}
                .map-type-control select {{
                    padding: 5px;
                }}
            </style>
        </head>
        <body>
            <div id="map"></div>
            <div id="modal" class="modal">
                <span class="close">&times;</span>
                <img class="modal-content" id="modalImg">
            </div>
            <script>
                var map = L.map('map').setView([{center_lat}, {center_lon}], {self.map_zoom});

                // 地理院地図の標準地図タイル
                var standardGSI = L.tileLayer('https://cyberjapandata.gsi.go.jp/xyz/std/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '<a href="https://maps.gsi.go.jp/development/ichiran.html">地理院タイル</a>',
                    maxZoom: 18
                }});

                // 地理院地図の淡色地図タイル
                var paleGSI = L.tileLayer('https://cyberjapandata.gsi.go.jp/xyz/pale/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '<a href="https://maps.gsi.go.jp/development/ichiran.html">地理院タイル</a>',
                    maxZoom: 18
                }});

                // 地理院地図の写真タイル
                var photoGSI = L.tileLayer('https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{{z}}/{{x}}/{{y}}.jpg', {{
                    attribution: '<a href="https://maps.gsi.go.jp/development/ichiran.html">地理院タイル</a>',
                    maxZoom: 18
                }});

                // OpenStreetMapタイル（バックアップ用）
                var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }});

                // デフォルトの地図として標準地図を設定
                standardGSI.addTo(map);

                // 地図切り替えコントロールを追加
                var mapTypeControl = L.control({{position: 'topright'}});
                mapTypeControl.onAdd = function(map) {{
                    var div = L.DomUtil.create('div', 'map-type-control');
                    div.innerHTML = `
                        <select onchange="changeMapType(this.value)">
                            <option value="standard">標準地図</option>
                            <option value="pale">淡色地図</option>
                            <option value="photo">写真</option>
                            <option value="osm">OpenStreetMap</option>
                        </select>
                    `;
                    return div;
                }};
                mapTypeControl.addTo(map);

                // 地図切り替え関数
                function changeMapType(type) {{
                    map.eachLayer(function(layer) {{
                        if(layer instanceof L.TileLayer) {{
                            map.removeLayer(layer);
                        }}
                    }});

                    switch(type) {{
                        case 'standard':
                            standardGSI.addTo(map);
                            break;
                        case 'pale':
                            paleGSI.addTo(map);
                            break;
                        case 'photo':
                            photoGSI.addTo(map);
                            break;
                        case 'osm':
                            osm.addTo(map);
                            break;
                    }}
                }}

                // 以下、モーダル関連の機能は変更なし
                var modal = document.getElementById('modal');
                var modalImg = document.getElementById('modalImg');
                var span = document.getElementsByClassName('close')[0];

                function showModal(imgSrc) {{
                    modal.style.display = 'block';
                    modalImg.src = imgSrc;
                    debugImageLoad(modalImg, imgSrc);
                }}

                span.onclick = function() {{
                    modal.style.display = 'none';
                }}

                window.onclick = function(event) {{
                    if (event.target == modal) {{
                        modal.style.display = 'none';
                    }}
                }}

                document.addEventListener('keydown', function(event) {{
                    if (event.key === "Escape") {{
                        modal.style.display = 'none';
                    }}
                }});

                // 軌跡の座標
                var coordinates = {matched_df[['latitude', 'longitude']].values.tolist()};

                // 軌跡を描画
                L.polyline(coordinates, {{
                    color: 'blue',
                    weight: 3,
                    opacity: 0.8
                }}).addTo(map);
        """

        # マーカーとポップアップを追加
        for idx, row in matched_df.iterrows():
            # 絶対パスから相対パスを作成
            frame_abs_path = os.path.abspath(row['frame_path'])
            frame_rel_path = os.path.relpath(frame_abs_path, output_dir)

            # Windowsパスを/に変換
            frame_rel_path = frame_rel_path.replace('\\', '/')

            print(f"Debug - Frame path for point {idx}:")
            print(f"  Absolute: {frame_abs_path}")
            print(f"  Relative: {frame_rel_path}")

            popup_content = f"""
                var popup{idx} = L.popup({{maxWidth: 200}})
                    .setContent(`<div style="text-align: center;">
                        <img src="{frame_rel_path}" class="custom-popup-image" 
                             onclick="showModal('{frame_rel_path}')" 
                             alt="Frame {idx}"
                             onload="console.log('Popup image {idx} loaded successfully')"
                             onerror="console.error('Error loading popup image {idx}')"
                        />
                        <br>
                        <small>Time: {row['timestamp']}<br>
                        Speed: {row['speed']:.1f} m/s</small>
                        <div class="debug-info"></div>
                    </div>`);

                L.circleMarker([{row['latitude']}, {row['longitude']}], {{
                    radius: 5,
                    color: 'blue',
                    fillColor: 'blue',
                    fillOpacity: 0.2
                }})
                .bindPopup(popup{idx})
                .addTo(map);
            """

            html_content += popup_content

            # オリジナルのGPS位置を表示
            if 'original_lat' in row and 'original_lon' in row:
                original_popup = f"""
                    L.circleMarker([{row['original_lat']}, {row['original_lon']}], {{
                        radius: 3,
                        color: 'red',
                        fillColor: 'red',
                        fillOpacity: 0.2
                    }})
                    .bindPopup("Original GPS: {row['timestamp']}<br>Speed: {row['speed']:.1f} m/s")
                    .addTo(map);
                """
                html_content += original_popup

        # HTMLを完成させる
        html_content += """
            // デバッグ情報の表示切り替え
            document.addEventListener('keypress', function(e) {
                if (e.key === 'd') {
                    let debugInfos = document.getElementsByClassName('debug-info');
                    for (let info of debugInfos) {
                        info.style.display = info.style.display === 'none' ? 'block' : 'none';
                    }
                }
            });
            </script>
        </body>
        </html>
        """

        # ファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Map saved to {output_path}")
        print("\nDebug information:")
        print(f"- Map file directory: {output_dir}")
        print(f"- Total frames to display: {len(matched_df)}")
        print("\nIf images are not displaying:")
        print("1. Check browser's developer console (F12) for errors")
        print("2. Press 'd' key to toggle debug information")
        print("3. Verify that the frames directory is accessible from the HTML file location")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create a map with Gopro frames and GPS data")
    parser.add_argument("video_path", help="Path to Gopro video file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Frame extraction interval in seconds (default: 5.0)")
    parser.add_argument("--radius", type=float, default=50,
                        help="Road matching search radius in meters (default: 50)")
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float,
                        help="Duration to process in seconds (default: entire video)")
    parser.add_argument("--zoom", type=int, default=17,
                        help="Initial map zoom level (default: 17)")
    parser.add_argument("--lat", type=float,
                        help="Latitude (only needed if GPS data is not available in the video)")
    parser.add_argument("--lon", type=float,
                        help="Longitude (only needed if GPS data is not available in the video)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of frames to process in each batch (default: 1000)")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # OpenCVの読み込み試行回数を増やす
    os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '10000'

    matcher = GoproMapMatcher()
    matcher.frame_interval = args.interval
    matcher.search_radius = args.radius
    matcher.start_time = args.start
    matcher.duration = args.duration
    matcher.map_zoom = args.zoom

    try:
        print("\n=== Processing Parameters ===")
        print(f"Video file: {args.video_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Frame interval: {args.interval} seconds")
        print(f"Start time: {args.start} seconds")
        print(f"Duration: {args.duration if args.duration else 'entire video'} seconds")
        print(f"Road matching radius: {args.radius} meters")
        print("===========================\n")

        # GPS情報の抽出と処理
        matched_df = matcher.extract_frames_and_gps(args.video_path, args.output_dir)

        if matched_df is None and (args.lat is not None and args.lon is not None):
            print(f"Using provided GPS coordinates: {args.lat}, {args.lon}")
            matched_df = matcher.extract_frames_and_gps(args.video_path, args.output_dir, args.batch_size)

        if matched_df is None:
            print("Error: No GPS data available and no coordinates provided")
            return

        print("\n=== Processing Results ===")
        print(f"Extracted frames: {len(matched_df)}")
        print(f"Time span: {(matched_df['timestamp'].max() - matched_df['timestamp'].min()).total_seconds():.1f} seconds")
        print(f"GPS location (center): {matched_df['latitude'].mean():.6f}, {matched_df['longitude'].mean():.6f}")
        print("========================\n")

        # 道路ネットワークの取得と位置のマッチング
        road_edges = matcher.get_road_network(matched_df)
        matched_df = matcher.match_to_roads(matched_df, road_edges)

        # 結果の保存
        output_map = os.path.join(args.output_dir, 'map.html')
        matcher.create_map(matched_df, output_map)
        csv_path = os.path.join(args.output_dir, 'matched_points.csv')
        matched_df.to_csv(csv_path, index=False)

        print("\n=== Files Generated ===")
        print(f"Map file: {output_map}")
        print(f"CSV file: {csv_path}")
        print(f"Frames directory: {os.path.join(args.output_dir, 'frames')}")
        print("=====================\n")

        print("Processing completed successfully!")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())