import pymysql
from datetime import datetime
import json
import numpy as np

def get_db_connection():
    return pymysql.connect(
        host='192.168.221.128',
        user='root',
        passwd='gmlrla',
        db='wisar_db',
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

class SimulationDB:
    def __init__(self):
        self.setup_database()

    def setup_database(self):
        """데이터베이스와 테이블 설정"""
        conn = None  # conn 변수를 먼저 선언
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                # 시뮬레이션 메타데이터 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS simulation_meta (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        simulation_date DATETIME,
                        age_group VARCHAR(20),
                        gender VARCHAR(10),
                        health_status VARCHAR(10),
                        num_simulations INT,
                        max_steps INT,
                        start_point_x FLOAT,
                        start_point_y FLOAT,
                        average_reward FLOAT,
                        total_time INT
                    )
                """)

                # 개별 경로 결과 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS path_results (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        simulation_id INT,
                        path_number INT,
                        path_points JSON,
                        reward FLOAT,
                        num_steps INT,
                        FOREIGN KEY (simulation_id) REFERENCES simulation_meta(id)
                    )
                """)

            conn.commit()
            print("Database tables created successfully")

        except Exception as e:
            print(f"Error setting up database: {e}")
        finally:
            if conn is not None:  # conn이 존재할 때만 close 호출
                conn.close()

    def save_simulation_results(self, paths, rewards, agent_params, start_pos, max_steps, execution_time):
        """시뮬레이션 결과 저장"""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                # 메타데이터 저장
                meta_query = """
                    INSERT INTO simulation_meta (
                        simulation_date, age_group, gender, health_status,
                        num_simulations, max_steps, start_point_x, start_point_y,
                        average_reward, total_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                meta_values = (
                    datetime.now(),
                    agent_params['age_group'],
                    agent_params['gender'],
                    agent_params['health_status'],
                    len(paths),
                    max_steps,
                    float(start_pos[0]),
                    float(start_pos[1]),
                    float(np.mean(rewards)),
                    execution_time
                )
                cursor.execute(meta_query, meta_values)
                simulation_id = cursor.lastrowid

                # 개별 경로 결과 저장
                path_query = """
                    INSERT INTO path_results (
                        simulation_id, path_number, path_points,
                        reward, num_steps
                    ) VALUES (%s, %s, %s, %s, %s)
                """
                for i, (path, reward) in enumerate(zip(paths, rewards)):
                    path_points = json.dumps([(float(x), float(y)) for x, y in path])
                    path_values = (
                        simulation_id,
                        i + 1,
                        path_points,
                        float(reward),
                        len(path)
                    )
                    cursor.execute(path_query, path_values)

                conn.commit()
                print(f"Simulation results saved with ID: {simulation_id}")
                return simulation_id

        except Exception as e:
            print(f"Error saving results: {e}")
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_simulation_results(self, simulation_id=None, limit=10):
        """시뮬레이션 결과 조회"""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                if simulation_id:
                    # 특정 시뮬레이션 결과 조회
                    cursor.execute("""
                        SELECT m.*, GROUP_CONCAT(p.reward) as rewards
                        FROM simulation_meta m
                        LEFT JOIN path_results p ON m.id = p.simulation_id
                        WHERE m.id = %s
                        GROUP BY m.id
                    """, (simulation_id,))
                else:
                    # 최근 시뮬레이션 결과 목록
                    cursor.execute("""
                        SELECT m.*, COUNT(p.id) as num_paths, AVG(p.reward) as avg_reward
                        FROM simulation_meta m
                        LEFT JOIN path_results p ON m.id = p.simulation_id
                        GROUP BY m.id
                        ORDER BY m.simulation_date DESC
                        LIMIT %s
                    """, (limit,))

                results = cursor.fetchall()
                return results

        except Exception as e:
            print(f"Error retrieving results: {e}")
            return None
        finally:
            if conn is not None:
                conn.close()

    def get_path_details(self, simulation_id):
        """특정 시뮬레이션의 경로 상세 정보 조회"""
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT path_number, path_points, reward, num_steps
                    FROM path_results
                    WHERE simulation_id = %s
                    ORDER BY path_number
                """, (simulation_id,))

                paths = cursor.fetchall()
                for path in paths:
                    path['path_points'] = json.loads(path['path_points'])
                return paths

        except Exception as e:
            print(f"Error retrieving path details: {e}")
            return None
        finally:
            if conn is not None:
                conn.close()