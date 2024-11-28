import os
import sys
import torch
from env import CustomGeoDataset, Environment
from agent import PointerDQN
from config import *

def initialize(mode='simulation'):
    """
    환경과 에이전트 초기화
    mode: 'simulation' 또는 'training'
    """
    print("\n=== 시스템 초기화 중 ===")
    
    # 필요한 디렉토리 생성
    os.makedirs('weights', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        dataset = CustomGeoDataset(
            dem_file=DEM_FILE,
            road_file=ROAD_FILE,
            forestroad_file=FORESTROAD_FILE,
            climbpath_file=CLIMBPATH_FILE
        )
        
        # 환경 초기화
        env = Environment(dataset, area_difference_file)
        
        # PointerDQN 에이전트 초기화
        agent = PointerDQN(
            input_dim=8,
            hidden_dim=128,
            output_dim=8,
            learning_rate=CONFIG['learning_rate']
        )
        
        if mode == 'simulation':
            # 시뮬레이션 모드에서는 학습된 모델 필요
            model_file = os.path.join('weights', 'best_model_pointer.pth')
            if not os.path.exists(model_file):
                print("오류: 학습된 모델 파일(best_model_pointer.pth)을 찾을 수 없습니다.")
                print("먼저 학습을 진행하거나, weights 폴더에 학습된 모델 파일을 넣어주세요.")
                sys.exit(1)
            
            # 학습된 모델 로드
            print(f"학습된 모델을 로드합니다: {model_file}")
            try:
                checkpoint = torch.load(model_file)
                if 'network_state_dict' in checkpoint:
                    agent.network.load_state_dict(checkpoint['network_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    agent.network.load_state_dict(checkpoint['model_state_dict'])
                print("모델 로드 완료")
            except Exception as e:
                print(f"오류: 모델 로드 실패 - {e}")
                print("프로그램을 종료합니다.")
                sys.exit(1)
        
        elif mode == 'training':
            # 학습 모드에서는 기존 모델 파일 여부 확인
            model_file = os.path.join('weights', 'best_model_pointer.pth')
            if os.path.exists(model_file):
                print("\n기존 학습 모델이 발견되었습니다.")
                while True:
                    choice = input("기존 모델을 이어서 학습하시겠습니까? (y/n): ").lower()
                    if choice == 'y':
                        try:
                            checkpoint = torch.load(model_file)
                            if 'network_state_dict' in checkpoint:
                                agent.network.load_state_dict(checkpoint['network_state_dict'])
                            elif 'model_state_dict' in checkpoint:
                                agent.network.load_state_dict(checkpoint['model_state_dict'])
                            print("기존 모델을 로드했습니다. 이어서 학습을 진행합니다.")
                        except Exception as e:
                            print(f"모델 로드 중 오류 발생: {e}")
                            print("새로운 모델로 학습을 시작합니다.")
                        break
                    elif choice == 'n':
                        print("새로운 모델로 학습을 시작합니다.")
                        break
                    else:
                        print("'y' 또는 'n'을 입력해주세요.")
            else:
                print("새로운 모델로 학습을 시작합니다.")
                
    except Exception as e:
        print(f"오류: 초기화 실패 - {e}")
        print("프로그램을 종료합니다.")
        sys.exit(1)
    
    return env, agent