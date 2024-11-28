from pyproj import CRS, Transformer
import numpy as np
import os, time, warnings
from colorama import init, Fore, Style
from agent import PointerDQN
from simulation import initialize, simulate, get_simulation_params
from training.trainer import train_pointer_dqn
from utils.coordinates import latlon_to_pixel
from config import *

warnings.filterwarnings('ignore', message='The Shapely GEOS version')
transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)

env = None
agent = None

def print_header():
    """프로그램 헤더 출력"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Fore.CYAN}================================================")
    print(f"         조난자 이동 경로 예측 시스템 ")
    print(f"================================================{Style.RESET_ALL}")
    print(f"\n시스템 구성요소 초기화 중...")

def print_loading_message():
    """로딩 메시지 출력"""
    loading_messages = [
        "지리 데이터 로딩 중...",
        "신경망 초기화 중...",
        "환경 설정 중...",
        "시뮬레이션 구성요소 준비 중...",
        "시스템 준비 완료!"
    ]
    
    for msg in loading_messages:
        print(f"{Fore.GREEN}> {msg}{Style.RESET_ALL}")
        time.sleep(0.5)
    print()

def print_menu():
    """메인 메뉴 출력"""
    print(f"{Fore.YELLOW}사용 가능한 명령어:{Style.RESET_ALL}")
    print("1. 새로운 모델 학습")
    print("2. 시뮬레이션 실행")
    print("3. 프로그램 종료")
    print("\n선택해주세요 (1-3): ", end='')

def main():
    global env, agent

    init()  # colorama 초기화
    print_header()
    print_loading_message()
    
    try:
        env, agent = initialize()
        print(f"{Fore.GREEN}시스템이 성공적으로 초기화되었습니다!{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}시스템 초기화 오류: {str(e)}{Style.RESET_ALL}")
        return
    
    while True:
        print_menu()
        choice = input().strip()
        
        if choice == '1':
            print(f"\n{Fore.CYAN}[학습 모드]{Style.RESET_ALL}")
            print("학습 프로세스를 시작합니다...")
            try:
                # 학습 모드로 초기화
                env, agent = initialize(mode='training')
                
                # 학습 실행
                trained_agent = train_pointer_dqn(env, agent, CONFIG)
                agent = trained_agent
                print(f"\n{Fore.GREEN}학습이 성공적으로 완료되었습니다!{Style.RESET_ALL}")
                
                print("\n학습 결과:")
                print(f"- weights/ 폴더에 모델이 저장되었습니다")
                print(f"- logs/ 폴더에 학습 로그가 저장되었습니다")
                print(f"- best_model_pointer.pth: 최고 성능 모델")
                print(f"- final_model_pointer.pth: 최종 모델")
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}학습이 사용자에 의해 중단되었습니다.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}학습 중 오류 발생: {str(e)}{Style.RESET_ALL}")

        elif choice == '2':
            print(f"\n{Fore.CYAN}[시뮬레이션 모드]{Style.RESET_ALL}")
            
            try:
                env, agent = initialize(mode='simulation')
                # 시뮬레이션 파라미터 설정
                params = get_simulation_params()
            
                print("\n시작 위치 선택 방법:")
                print("1. 무작위 위치")
                print("2. 수동 좌표 입력")
                sim_choice = input("선택해주세요 (1-2): ").strip()
                
                if sim_choice == '1':
                    print("\n무작위 위치 생성 중...")
                    try:
                        start_pos = env.get_random_start_point()
                        utm_coords = env.transform * (start_pos[0] + 0.5, start_pos[1] + 0.5)
                        utm_x, utm_y = float(utm_coords[0]), float(utm_coords[1])
                        result = transformer.transform(utm_x, utm_y)
                        lon, lat = float(result[0]), float(result[1])
        
                        print(f"{Fore.GREEN}무작위 위치가 성공적으로 생성되었습니다:{Style.RESET_ALL}")
                        print(f"픽셀 좌표: ({start_pos[0]}, {start_pos[1]})")
                        print(f"지리 좌표: ({lat:.6f}, {lon:.6f})")
                        print("\n시뮬레이션을 시작합니다...")
                        
                        paths, rewards = simulate(env, agent, 
                                            agent_params=params['agent_params'],
                                            num_simulations=params['num_simulations'],
                                            max_steps=params['max_steps'],
                                            start_pos=start_pos)
                        
                        if paths and rewards:
                            print(f"\n{Fore.GREEN}시뮬레이션이 성공적으로 완료되었습니다!{Style.RESET_ALL}")
                            print(f"생성된 경로 수: {len(paths)}")
                            print(f"평균 보상: {np.mean(rewards):.2f}")
                        else:
                            print(f"{Fore.RED}경로 생성에 실패했습니다.{Style.RESET_ALL}")
                            
                    except Exception as e:
                        print(f"{Fore.RED}무작위 위치 생성 오류: {str(e)}{Style.RESET_ALL}")
                        
                elif sim_choice == '2':
                    print("\n좌표를 입력해주세요:")
                    try:
                        lat = float(input("위도 (예: 35.123456): "))
                        lon = float(input("경도 (예: 128.123456): "))
                        print("\n좌표 유효성 검사 중...")
                        
                        try:
                            x, y = latlon_to_pixel(lat, lon, env.transform, env.crs)
                            
                            if (0 <= x < env.shape[1] and 0 <= y < env.shape[0] and
                                env.area_mask[y, x] == 1 and not env._is_in_water((x, y))):
                                start_pos = (x, y)
                                print(f"{Fore.GREEN}좌표가 성공적으로 검증되었습니다!{Style.RESET_ALL}")
                                print(f"픽셀 좌표로 변환: ({x}, {y})")
                                
                                paths, rewards = simulate(env, agent, 
                                                    agent_params=params['agent_params'],
                                                    num_simulations=params['num_simulations'],
                                                    max_steps=params['max_steps'],
                                                    start_pos=start_pos)
                                
                                if paths and rewards:
                                    print(f"\n{Fore.GREEN}시뮬레이션이 성공적으로 완료되었습니다!{Style.RESET_ALL}")
                                    print(f"생성된 경로 수: {len(paths)}")
                                    print(f"평균 보상: {np.mean(rewards):.2f}")
                                else:
                                    print(f"{Fore.RED}경로 생성에 실패했습니다.{Style.RESET_ALL}")
                            else:
                                print(f"{Fore.RED}잘못된 위치: 유효 영역을 벗어났거나 수계 지역입니다{Style.RESET_ALL}")
                        except Exception as e:
                            print(f"{Fore.RED}좌표 변환 오류: {str(e)}{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}잘못된 좌표입니다. 십진수 형식으로 입력해주세요.{Style.RESET_ALL}")
            
            except Exception as e:
                print(f"{Fore.Red}시뮬레이션 오류 : {str(e)}{Style.RESET_ALL}")
            
        elif choice == '3':
            print("\n시스템을 종료합니다...")
            time.sleep(1)
            print(f"{Fore.GREEN}프로그램이 성공적으로 종료되었습니다.{Style.RESET_ALL}")
            break
            
        else:
            print(f"{Fore.RED}잘못된 선택입니다. 다시 시도해주세요.{Style.RESET_ALL}")
        
        print("\n계속하려면 Enter를 눌러주세요...")
        input()
        os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    main()