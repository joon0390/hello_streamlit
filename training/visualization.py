import folium
import geopandas as gpd
import numpy as np
import os, json, random, webbrowser
import torch
from datetime import datetime

from pyproj import Transformer
from  utils.coordinates import pixel_to_latlon
from config import *

transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)

def create_visualization(env, paths, total_rewards, agent_params, output_file='simulation_results.html'):
    """Create visualization of the paths with improved styling and smoothing"""

    # 경로 저장 디렉토리 생성
    results_dir = 'path_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    
    age_str = agent_params['age_group']
    gender_str = agent_params['gender']
    health_str = 'healthy' if agent_params['health_status'] == 'good' else 'sick'

    # 현재 시간을 포함한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'simulation_{age_str}_{gender_str}_{health_str}_{timestamp}'
    
    # HTML 파일과 JSON 파일 경로 설정
    html_file = os.path.join(results_dir, f'{filename}.html')
    json_file = os.path.join(results_dir, f'{filename}_info.json')


    def smooth_path(coordinates, smoothing_factor=0.5, iterations=3):
        """경로 스무딩 함수"""
        if len(coordinates) <= 2:
            return coordinates

        smoothed = list(coordinates)
        for _ in range(iterations):
            temp = list(smoothed)
            for i in range(1, len(coordinates) - 1):
                for j in range(2):
                    prev = smoothed[i-1][j]
                    curr = smoothed[i][j]
                    next_val = smoothed[i+1][j]
                    new_val = curr + smoothing_factor * ((prev + next_val) / 2 - curr)
                    temp[i][j] = new_val
            smoothed = temp
        return smoothed

    # 모든 경로의 좌표를 GIS 좌표계로 변환
    paths_gis = []
    for path in paths:
        path_coords = []
        for x, y in path:
            utm_coords = env.dataset.transform * (x, y)
            utm_x, utm_y = float(utm_coords[0]), float(utm_coords[1])
            result = transformer.transform(utm_x, utm_y)
            lon, lat = float(result[0]), float(result[1])
            path_coords.append([lon, lat])
        paths_gis.append(path_coords)

    # 중심점 계산
    all_coords = [coord for path in paths_gis for coord in path]
    center_lat = np.mean([coord[1] for coord in all_coords])
    center_lon = np.mean([coord[0] for coord in all_coords])

    # 기본 맵 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # 도로 데이터 로드 및 추가
    road_data = gpd.read_file(ROAD_FILE)
    forestroad_data = gpd.read_file(FORESTROAD_FILE)
    climbpath_data = gpd.read_file(CLIMBPATH_FILE)

    # 도로 스타일 정의
    road_style = lambda x: {'color': '#FF0000', 'weight': 2}
    forestroad_style = lambda x: {'color': '#00FF00', 'weight': 2}
    climbpath_style = lambda x: {'color': '#0000FF', 'weight': 2}

    # 도로 레이어 추가
    folium.GeoJson(
        road_data,
        name='Roads',
        style_function=road_style,
        tooltip="Main Road"
    ).add_to(m)

    folium.GeoJson(
        forestroad_data,
        name='Forest Roads',
        style_function=forestroad_style,
        tooltip="Forest Road"
    ).add_to(m)

    folium.GeoJson(
        climbpath_data,
        name='Climbing Paths',
        style_function=climbpath_style,
        tooltip="Climbing Path"
    ).add_to(m)

    # 경로별 색상 설정 (파란색 계열)
    colors = ['#cae9fd', '#97d3f9', '#65bdf5', '#32a7f1', '#008cec']

    # 시뮬레이션 경로 추가
    for i, (path_gis, reward) in enumerate(zip(paths_gis, total_rewards)):
        if not path_gis:
            continue

        # 색상 선택
        color = colors[int(i * (len(colors)-1) / max(len(paths_gis)-1, 1))]

        # 시작점 마커
        folium.Marker(
            location=[path_gis[0][1], path_gis[0][0]],
            icon=folium.Icon(color='green', icon='flag', prefix='fa'),
            popup=f'Start Point of Path {i+1}'
        ).add_to(m)

        # 끝점 마커
        folium.Marker(
            location=[path_gis[-1][1], path_gis[-1][0]],
            icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa'),
            popup=f'End Point of Path {i+1}'
        ).add_to(m)

        # 좌표 변환 및 스무딩
        coordinates = [[coord[1], coord[0]] for coord in path_gis]
        smoothed_coordinates = smooth_path(coordinates)

        # 원본 경로 (흐리게)
        folium.PolyLine(
            locations=coordinates,
            weight=2,
            color=color,
            opacity=0.3,
            popup=f'Original Path {i+1}'
        ).add_to(m)

        # 스무딩된 경로
        folium.PolyLine(
            locations=smoothed_coordinates,
            weight=3,
            color=color,
            popup=f'Path {i+1} (Reward: {reward:.2f})',
            opacity=0.8
        ).add_to(m)

        # 경로 정보 팝업
        info_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Path {i+1}</b><br>
            Steps: {len(coordinates)}<br>
            Reward: {reward:.2f}<br>
            Start: ({path_gis[0][1]:.6f}, {path_gis[0][0]:.6f})<br>
            End: ({path_gis[-1][1]:.6f}, {path_gis[-1][0]:.6f})
        </div>
        """
        folium.Popup(info_html).add_to(
            folium.Marker(
                location=[path_gis[0][1], path_gis[0][0]],
                icon=folium.DivIcon(html=f'<div style="color: {color};">●</div>')
            )
        )

    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)
    m.save(html_file)
    print(f"\n지도 파일이 저장되었습니다: {html_file}")

    results_info = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'agent_info': {
            'age_group': age_str,
            'gender': gender_str,
            'health_status': health_str
        },
        'simulation_stats': {
            'num_paths': len(paths),
            'total_rewards': [float(r) for r in total_rewards],
            'average_reward': float(np.mean(total_rewards)) if total_rewards else 0,
            'max_reward': float(max(total_rewards)) if total_rewards else 0,
            'min_reward': float(min(total_rewards)) if total_rewards else 0,
            'paths_length': [len(path) for path in paths]
        },
        'visualization_file': html_file
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_info, f, ensure_ascii=False, indent=2)
    
    print(f"결과 정보가 저장되었습니다: {json_file}")
    #webbrowser.open('file://' + os.path.realpath(html_file))

def get_action_with_diversity(q_values, epsilon_second=0.3, epsilon_third=0.2):
    """
    Q-values에서 확률적으로 top-3 행동 중 하나를 선택
    Args:
        q_values: 모델이 출력한 Q-values
        epsilon_second: 두 번째로 좋은 행동을 선택할 확률
        epsilon_third: 세 번째로 좋은 행동을 선택할 확률
    """
    sorted_actions = torch.argsort(q_values, descending=True)
    rand_val = random.random()
    
    if rand_val < epsilon_second and len(sorted_actions) >= 2:
        return sorted_actions[1].item()  # 두 번째로 좋은 행동
    elif rand_val < (epsilon_second + epsilon_third) and len(sorted_actions) >= 3:
        return sorted_actions[2].item()  # 세 번째로 좋은 행동
    else:
        return sorted_actions[0].item()  # 가장 좋은 행동