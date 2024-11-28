import streamlit as st
import folium
import torch
import time
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
import os, json
from pyproj import Transformer
from utils.initialization import initialize
from utils.coordinates import latlon_to_pixel
from config import *
from simulation import simulate
from db_manager import SimulationDB

# 전역 변수 정의
transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
age_map = {'young': '20대', 'middle': '40-50대', 'old': '60대 이상'}
gender_map = {'male': '남성', 'female': '여성'}
health_map = {'good': '양호', 'bad': '나쁨'}

def smooth_path(coordinates, smoothing_factor=0.5, iterations=3):
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

def create_visualization(env, paths, total_rewards, agent_params):
    """Create visualization of the paths and return folium map object"""   
    # GIS 좌표로 변환
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

    # 도로 레이어 추가
    folium.GeoJson(
        road_data,
        name='일반도로',
        style_function=lambda x: {'color': '#FF0000', 'weight': 2},
        tooltip="일반도로"
    ).add_to(m)

    folium.GeoJson(
        forestroad_data,
        name='임도',
        style_function=lambda x: {'color': '#00FF00', 'weight': 2},
        tooltip="임도"
    ).add_to(m)

    folium.GeoJson(
        climbpath_data,
        name='등산로',
        style_function=lambda x: {'color': '#0000FF', 'weight': 2},
        tooltip="등산로"
    ).add_to(m)

    # 경로 시각화
    colors = ['#cae9fd', '#97d3f9', '#65bdf5', '#32a7f1', '#008cec']
    for i, (path_gis, reward) in enumerate(zip(paths_gis, total_rewards)):
        if not path_gis:
            continue

        color = colors[int(i * (len(colors)-1) / max(len(paths_gis)-1, 1))]

        # 시작/종료 마커 추가
        folium.Marker(
            location=[path_gis[0][1], path_gis[0][0]],
            icon=folium.Icon(color='green', icon='flag', prefix='fa'),
            popup=f'시작 지점 (경로 {i+1})'
        ).add_to(m)

        folium.Marker(
            location=[path_gis[-1][1], path_gis[-1][0]],
            icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa'),
            popup=f'종료 지점 (경로 {i+1})'
        ).add_to(m)

        # 경로 그리기
        coordinates = [[coord[1], coord[0]] for coord in path_gis]
        smoothed_coordinates = smooth_path(coordinates)

        # 원본 경로
        folium.PolyLine(
            locations=coordinates,
            weight=2,
            color=color,
            opacity=0.3,
            popup=f'원본 경로 {i+1}'
        ).add_to(m)

        # 스무딩된 경로
        folium.PolyLine(
            locations=smoothed_coordinates,
            weight=3,
            color=color,
            popup=f'경로 {i+1} (보상: {reward:.2f})',
            opacity=0.8
        ).add_to(m)

        # 경로 정보
        info_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>경로 {i+1}</b><br>
            이동 거리: {len(coordinates)} 스텝<br>
            보상: {reward:.2f}<br>
            시작: ({path_gis[0][1]:.6f}, {path_gis[0][0]:.6f})<br>
            종료: ({path_gis[-1][1]:.6f}, {path_gis[-1][0]:.6f})
        </div>
        """
        folium.Popup(info_html).add_to(
            folium.Marker(
                location=[path_gis[0][1], path_gis[0][0]],
                icon=folium.DivIcon(html=f'<div style="color: {color};">●</div>')
            )
        )

    folium.LayerControl().add_to(m)
    return m

def main():
    st.set_page_config(page_title="⛰️ 조난자 이동 경로 예측 시스템", layout="wide")
    
    # 세션 상태 초기화
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'map_object' not in st.session_state:
        st.session_state.map_object = None  

    st.title("⛰️ 조난자 이동 경로 예측 시스템")
    
    with st.sidebar:
        st.header("시뮬레이션 설정 🛠️")
        
        st.subheader("조난자 정보 🧍🏻")
        age_group = st.selectbox(
            "나이대",
            options=["20대", "40-50대", "60대 이상"],
            format_func=lambda x: {"20대": "👦🏻 20대", "40-50대": "👨🏻 40-50대", "60대 이상": "👨🏻‍🦳 60대 이상"}[x]
        )
        
        gender = st.radio("성별", ["남성 ⚨", "여성 ♀"])
        health_status = st.radio("건강상태", ["양호", "나쁨"])
        
        st.subheader("시뮬레이션 설정 ⚙️")
        num_simulations = st.number_input("시뮬레이션 횟수", min_value=1, max_value=20, value=3)
        max_steps = st.number_input("최대 시뮬레이션 시간(초)", min_value=100, max_value=2000, value=1000)
        
        st.subheader("시작 위치 설정 📍")
        start_pos_method = st.radio("시작 위치 선택 방법", ["무작위", "좌표 입력"])
        
        if start_pos_method == "좌표 입력":
            lat = st.number_input("위도", value=35.123456, format="%f")
            lon = st.number_input("경도", value=128.123456, format="%f")
        
        run_simulation = st.button("시뮬레이션 시작 🚀")
    
    col1, col2 = st.columns([2, 1])

    if run_simulation:
        try:
            env, agent = initialize(mode='simulation')
            db = SimulationDB()
            
            agent_params = {
                'age_group': 'young' if age_group == "20대" else 'middle' if age_group == "40-50대" else 'old',
                'gender': 'male' if "남성" in gender else 'female',
                'health_status': 'good' if health_status == "양호" else 'bad'
            }
            
            if start_pos_method == "무작위":
                start_pos = env.get_random_start_point()
                utm_x, utm_y = env.transform * (start_pos[0] + 0.5, start_pos[1] + 0.5)
                lon, lat = transformer.transform(utm_x, utm_y)
                st.info(f"생성된 시작 위치 - 위도: {lat:.6f}, 경도: {lon:.6f}")
            else:
                x, y = latlon_to_pixel(lat, lon, env.transform, env.crs)
                start_pos = (x, y)
            
            with st.spinner('시뮬레이션 실행 중... ⏳'):
                start_time = time.time()
                paths, rewards = simulate(
                    env=env,
                    agent=agent,
                    agent_params=agent_params,
                    num_simulations=num_simulations,
                    max_steps=max_steps,
                    start_pos=start_pos
                )
                execution_time = int(time.time() - start_time)
                
                # DB에 결과 저장
                simulation_id = db.save_simulation_results(
                    paths=paths,
                    rewards=rewards,
                    agent_params=agent_params,
                    start_pos=start_pos,
                    max_steps=max_steps,
                    execution_time=execution_time
                )
                
                # 결과를 세션 상태에 저장
                st.session_state.simulation_results = {
                    'id': simulation_id,
                    'paths': paths,
                    'rewards': rewards,
                    'agent_params': agent_params,
                    'execution_time': execution_time
                }
                
                # 지도 생성
                m = create_visualization(env, paths, rewards, agent_params)
                st.session_state.map_object = m
            
            st.success(f'시뮬레이션이 완료되었습니다! (ID: {simulation_id}) ✨')
            
        except Exception as e:
            st.error(f'오류가 발생했습니다: {str(e)} ❌')
    
    # 결과 표시
    with col1:
        st.subheader("🗺️ 시뮬레이션 결과 지도")
        if st.session_state.map_object is not None:
            st_folium(st.session_state.map_object, width=800, height=600)
    
    with col2:
        st.subheader("📊 시뮬레이션 결과")
        if st.session_state.simulation_results is not None:
            results = st.session_state.simulation_results
            
            # 기본 정보 표시
            st.markdown("### 🔍 기본 정보")
            st.write(f"시뮬레이션 ID: {results['id']}")
            st.write(f"실행 시간: {results['execution_time']}초")
            
            agent_info = {
                '나이': age_map[results['agent_params']['age_group']],
                '성별': gender_map[results['agent_params']['gender']],
                '건강상태': health_map[results['agent_params']['health_status']]
            }
            st.write("조난자 정보:", agent_info)
            
            # 결과 테이블
            results_df = pd.DataFrame({
                '시뮬레이션': range(1, len(results['rewards']) + 1),
                '보상': results['rewards'],
                '경로 길이': [len(path) for path in results['paths']]
            })
            
            st.dataframe(results_df)
            
            # 통계
            st.markdown("### 📈 통계")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 보상", f"{np.mean(results['rewards']):.2f}")
            with col2:
                st.metric("최대 보상", f"{max(results['rewards']):.2f}")
            with col3:
                st.metric("최소 보상", f"{min(results['rewards']):.2f}")
    
    # 이전 결과 조회 섹션 추가
    st.subheader("📜 이전 시뮬레이션 결과")
    if st.button("이전 결과 조회"):
        db = SimulationDB()
        previous_results = db.get_simulation_results(limit=5)
        if previous_results:
            st.write("최근 5개 시뮬레이션 결과:")
            
            # 데이터프레임 변환 및 컬럼명 한글화
            results_df = pd.DataFrame(previous_results)
            if not results_df.empty:
                column_map = {
                    'id': '시뮬레이션 ID',
                    'simulation_date': '실행 시간',
                    'age_group': '나이',
                    'gender': '성별',
                    'health_status': '건강상태',
                    'num_simulations': '시뮬레이션 횟수',
                    'max_steps': '최대 스텝',
                    'average_reward': '평균 보상',
                    'total_time': '총 실행시간(초)'
                }
                results_df = results_df.rename(columns=column_map)
                
                # 나이, 성별, 건강상태 한글 변환
                results_df['나이'] = results_df['나이'].map(age_map)
                results_df['성별'] = results_df['성별'].map(gender_map)
                results_df['나이'] = results_df['나이'].map(age_map)
                results_df['성별'] = results_df['성별'].map(gender_map)
                results_df['건강상태'] = results_df['건강상태'].map(health_map)
                
                # 데이터프레임 표시
                st.dataframe(results_df.style.format({
                    '평균 보상': '{:.2f}',
                    '총 실행시간(초)': '{:.0f}'
                }))
                
                # 특정 시뮬레이션 상세 정보 조회
                if len(results_df) > 0:
                    selected_id = st.selectbox(
                        "상세 정보를 볼 시뮬레이션 선택", 
                        results_df['시뮬레이션 ID'].tolist(),
                        format_func=lambda x: f"시뮬레이션 #{x}"
                    )
                    if selected_id:
                        path_details = db.get_path_details(selected_id)
                        if path_details:
                            st.write("선택한 시뮬레이션의 경로 상세 정보:")
                            
                            # 경로 정보를 데이터프레임으로 변환
                            path_df = pd.DataFrame(path_details)
                            path_df = path_df.rename(columns={
                                'path_number': '경로 번호',
                                'reward': '보상',
                                'num_steps': '스텝 수'
                            })
                            path_df['보상'] = path_df['보상'].round(2)
                            
                            # path_points는 복잡한 JSON이므로 제외하고 표시
                            path_df = path_df.drop('path_points', axis=1)
                            st.dataframe(path_df)
                            
                            # 통계 표시
                            st.markdown("### 경로 통계")
                            avg_reward = path_df['보상'].mean()
                            avg_steps = path_df['스텝 수'].mean()
                            st.write(f"평균 보상: {avg_reward:.2f}")
                            st.write(f"평균 스텝 수: {avg_steps:.1f}")
        else:
            st.info("저장된 이전 결과가 없습니다.")

if __name__ == "__main__":
    main()