import os, json, random, time
import numpy as np
import torch
from training.visualization import create_visualization, get_action_with_diversity
from env import Environment
from utils.dataset import CustomGeoDataset
from agent import PointerDQN, Agent
from config import *
from utils.initialization import initialize  
from db_manager import SimulationDB


def simulate(env, agent, agent_params, num_simulations=3, max_steps=700, start_pos=None):
    """시뮬레이션 실행 함수"""
    # DB 연결
    db = SimulationDB()
    start_time = time.time()
    
    print(f"\n시뮬레이션을 시작합니다...")
    print(f"조난자 특성: {agent_params}")
    
    # 에이전트 네트워크를 eval 모드로 설정
    agent.network.eval()
    paths = []
    total_rewards = []
    
    if start_pos is None:
        raise ValueError("시작 위치를 지정해야 합니다")
    
    print(f"시작 위치: {start_pos}")
    
    for sim in range(num_simulations):
        print(f"\n시뮬레이션 {sim + 1}/{num_simulations} 시작...")
        current_path = [start_pos]
        total_reward = 0
        state = env.reset(start_pos)
        
        for step in range(max_steps):
            with torch.no_grad():
                q_values, _ = agent.network(torch.FloatTensor(state).unsqueeze(0))
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            current_path.append(env.current_position)
            
            if done:
                break
            state = next_state
        
        paths.append(current_path)
        total_rewards.append(total_reward)
        print(f"시뮬레이션 {sim + 1} 완료. 총 보상: {total_reward:.2f}, 스텝: {len(current_path)}")

    # 실행 시간 계산
    execution_time = int(time.time() - start_time)
    
    # DB에 결과 저장
    simulation_id = db.save_simulation_results(
        paths=paths,
        rewards=total_rewards,
        agent_params=agent_params,
        start_pos=start_pos,
        max_steps=max_steps,
        execution_time=execution_time
    )
    
    print(f"\n결과가 데이터베이스에 저장되었습니다. (Simulation ID: {simulation_id})")
    
    print("\n경로 지도 생성 중...")
    try:
        create_visualization(env, paths, total_rewards, agent_params)
    except Exception as e:
        print(f"지도 생성 중 오류 발생: {str(e)}")
        
    return paths, total_rewards


# simulation.py
def get_simulation_params():
    print("\n=== 시뮬레이션 설정 ===")
    
    # 시뮬레이션 횟수 입력
    while True:
        try:
            num_simulations = int(input("시뮬레이션 횟수를 입력하세요: ")) -1
            if num_simulations > 0:
                break
            print("1 이상의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    # 최대 스텝 수 입력 (초 단위)
    while True:
        try:
            max_steps = int(input("최대 시뮬레이션 시간을 입력하세요 (초 단위): "))
            if max_steps > 0:
                break
            print("1 이상의 숫자를 입력해주세요.")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
    
    # 조난자 특성 입력
    print("\n=== 조난자 특성 설정 ===")
    print("1. 나이대 선택:")
    print("  1) 20대")
    print("  2) 40-50대")
    print("  3) 60대 이상")
    while True:
        age_choice = input("선택 (1-3): ")
        if age_choice in ['1', '2', '3']:
            age_group = {'1': 'young', '2': 'middle', '3': 'old'}[age_choice]
            break
        print("1-3 중에서 선택해주세요.")
    
    print("\n2. 성별 선택:")
    print("  1) 남성")
    print("  2) 여성")
    while True:
        gender_choice = input("선택 (1-2): ")
        if gender_choice in ['1', '2']:
            gender = {'1': 'male', '2': 'female'}[gender_choice]
            break
        print("1-2 중에서 선택해주세요.")
    
    print("\n3. 건강상태 선택:")
    print("  1) 양호")
    print("  2) 나쁨")
    while True:
        health_choice = input("선택 (1-2): ")
        if health_choice in ['1', '2']:
            health_status = {'1': 'good', '2': 'bad'}[health_choice]
            break
        print("1-2 중에서 선택해주세요.")
    
    # 환경 조건 입력 (추후 개발용)
    print("\n=== 환경 조건 설정 (현재는 기본값으로 실행) ===")
    weather = input("날씨 (맑음/흐림/비/눈): ")
    time_of_day = input("시간대 (아침/낮/저녁/밤): ")
    season = input("계절 (봄/여름/가을/겨울): ")
    
    return {
        'num_simulations': num_simulations,
        'max_steps': max_steps,
        'agent_params': {
            'age_group': age_group,
            'gender': gender,
            'health_status': health_status
        },
        'env_params': {
            'weather': weather,
            'time_of_day': time_of_day,
            'season': season
        }
    }