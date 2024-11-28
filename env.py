import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import Point
from collections import deque
import random
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import reproject, Resampling, calculate_default_transform

from config import *
class CustomGeoDataset:
    def __init__(self, dem_file, road_file, forestroad_file, climbpath_file, 
                 rirsv_file=None, wkmstrm_file=None, watershed_file=None, channels_file=None):
        with rasterio.open(dem_file) as src:
            target_crs = CRS.from_epsg(32652)
            if src.crs != target_crs:
                transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
                dem_data = np.zeros((height, width), dtype=src.dtypes[0])
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dem_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )
                
                self.dem = dem_data
                self.transform = transform
                self.crs = target_crs
                self.shape = (height, width)
            else:
                self.dem = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                self.shape = src.shape

        self.road_data = self._vectorize_roads(road_file, "도로")
        self.forestroad_data = self._vectorize_roads(forestroad_file, "산림도로")
        self.climbpath_data = self._vectorize_roads(climbpath_file, "등산로")

        self.rirsv_data = self._vectorize_roads(rirsv_file, "저수지") if rirsv_file else np.zeros(self.shape)
        self.wkmstrm_data = self._vectorize_roads(wkmstrm_file, "하천") if wkmstrm_file else np.zeros(self.shape)
        self.watershed_data = self._vectorize_roads(watershed_file, "유역") if watershed_file else np.zeros(self.shape)
        self.channels_data = self._vectorize_roads(channels_file, "수로") if channels_file else np.zeros(self.shape)

        self.slope = self._calculate_slope()

    def _vectorize_roads(self, road_file, road_type="road"):
        try:
            roads = gpd.read_file(road_file)
            if roads.crs != self.crs:
                roads = roads.to_crs(self.crs)
            road_raster = rasterize(
                [(geom, 1) for geom in roads.geometry],
                out_shape=self.shape,
                transform=self.transform,
                fill=0,
                dtype=np.uint8
            )
            return road_raster
        except Exception as e:
            return np.zeros(self.shape, dtype=np.uint8)

    def _calculate_slope(self):
        dy, dx = np.gradient(self.dem)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        return slope

    def __getitem__(self, idx):
        x, y = idx
        patch_size = 3
        half_size = patch_size // 2
        
        x_start = max(0, x - half_size)
        x_end = min(self.shape[1], x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(self.shape[0], y + half_size + 1)
        
        dem_patch = self.dem[y_start:y_end, x_start:x_end]
        slope_patch = self.slope[y_start:y_end, x_start:x_end]
        road_patch = self.road_data[y_start:y_end, x_start:x_end]
        forestroad_patch = self.forestroad_data[y_start:y_end, x_start:x_end]
        climbpath_patch = self.climbpath_data[y_start:y_end, x_start:x_end]
        
        state = np.array([
            self.dem[y, x],
            self.slope[y, x],
            np.mean(dem_patch),
            np.mean(slope_patch),
            np.max(road_patch),
            np.max(forestroad_patch),
            np.max(climbpath_patch),
            np.std(dem_patch)
        ])
        
        return state

class Environment:
    def __init__(self, dataset, area_difference_file, max_timesteps=1000):
        self.dataset = dataset
        self.shape = dataset.shape
        self.transform = dataset.transform
        self.crs = dataset.crs
        
        # 환경 데이터 로드
        self.area_gdf = gpd.read_file(area_difference_file)
        if self.area_gdf.crs != self.crs:
            self.area_gdf = self.area_gdf.to_crs(self.crs)
        self.area_mask = self._rasterize_data(self.area_gdf, "area mask")
        self.load_environmental_data()
        
        # 상태 추적
        self.current_position = None
        self.visited = set()
        self.path = []  # 이동 경로 저장
        self.timestep = 0
        self.max_timesteps = max_timesteps
        self.following_path = False  # 길을 따라가는 중인지 여부
        
        # 행동 정의
        self.moves = {
            0: (-1, 0),   # Up
            1: (1, 0),    # Down
            2: (0, -1),   # Left
            3: (0, 1),    # Right
            4: (-1, -1),  # Up-Left
            5: (-1, 1),   # Up-Right
            6: (1, -1),   # Down-Left
            7: (1, 1)     # Down-Right
        }

    def get_random_start_point(self):
        """Get random start point within area_difference polygon"""
        while True:
            # Generate random point within DEM bounds
            random_x = np.random.randint(0, self.shape[1])
            random_y = np.random.randint(0, self.shape[0])
            
            # Convert pixel coordinates to UTM
            utm_x, utm_y = self.transform * (random_x + 0.5, random_y + 0.5)
            point = Point(utm_x, utm_y)
            
            # Check if point is within area_difference and not in water
            if (self.area_mask[random_y, random_x] == 1 and
                not self._is_in_water((random_x, random_y))):
                return random_x, random_y
            
    def _is_in_water(self, position):
        """Check if position is in water features"""
        x, y = position
        return (self.rirsv_data[y, x] == 1 or
                self.wkmstrm_data[y, x] == 1)
    
    def _is_near_water(self, x, y, radius=3):
        """Check if position is near water features"""
        x_start = max(0, x - radius)
        x_end = min(self.shape[1], x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(self.shape[0], y + radius + 1)
        
        water_patch = (self.rirsv_data[y_start:y_end, x_start:x_end] |
                      self.wkmstrm_data[y_start:y_end, x_start:x_end] |
                      self.channels_data[y_start:y_end, x_start:x_end])
        
        return np.any(water_patch)

    def _get_ranked_path_directions(self):
        """현재 위치에서 가능한 경로들을 점수 순으로 정렬하여 반환"""
        x, y = self.current_position
        available_moves = []
        
        # 초기 진행 방향 결정
        initial_direction = None
        if len(self.path) >= 3:
            first_pos = self.path[0]
            current_pos = self.current_position
            initial_vector = (current_pos[0] - first_pos[0], current_pos[1] - first_pos[1])
            if abs(initial_vector[0]) > abs(initial_vector[1]):
                initial_direction = 'horizontal'
            else:
                initial_direction = 'vertical'
        
        # 현재 진행 방향 계산
        current_direction = None
        if len(self.path) > 1:
            prev_x, prev_y = self.path[-2]
            current_direction = (x - prev_x, y - prev_y)
        
        # 8방향 검사 및 점수 계산
        for dy, dx in self.moves.values():
            new_y, new_x = y + dy, x + dx
            
            # 기본 검증
            if not (0 <= new_x < self.shape[1] and 0 <= new_y < self.shape[0]):
                continue
            if not self._is_on_path((new_x, new_y)):
                continue
            
            # 방문 횟수 확인
            visit_count = sum(1 for pos in self.path[-10:] if pos == (new_x, new_y))
            if visit_count > 1:
                continue
            
            # 새로운 위치의 고도
            current_elevation = self.dataset.dem[y, x]
            new_elevation = self.dataset.dem[new_y, new_x]
            elevation_change = abs(new_elevation - current_elevation)
            
            if elevation_change > 5:
                continue
            
            # 점수 계산
            score = 0.0
            
            # 1. 초기 방향 유지 보너스
            if initial_direction:
                if initial_direction == 'horizontal':
                    if abs(dx) > abs(dy):
                        score += 5.0
                else:
                    if abs(dy) > abs(dx):
                        score += 5.0
            
            # 2. 현재 방향 유지 보너스
            if current_direction:
                if (dx, dy) == current_direction:
                    score += 3.0
                elif abs(dx - current_direction[0]) + abs(dy - current_direction[1]) <= 1:
                    score += 1.5
            
            # 3. 도로 타입 보너스
            if self.dataset.road_data[new_y, new_x] == 1:
                score += 2.0
            elif self.dataset.forestroad_data[new_y, new_x] == 1:
                score += 1.5
            elif self.dataset.climbpath_data[new_y, new_x] == 1:
                score += 1.0
            
            # 4. 연결성 점수
            connected_paths = sum(1 for next_dy, next_dx in self.moves.values()
                                if (0 <= new_x + next_dx < self.shape[1] and 
                                    0 <= new_y + next_dy < self.shape[0] and 
                                    self._is_on_path((new_x + next_dx, new_y + next_dy))))
            score += connected_paths * 0.5
            
            # 5. 고도 변화 페널티
            score -= elevation_change * 0.2
            
            # 6. 방문 횟수 페널티
            score -= visit_count * 2.0
            
            available_moves.append({
                'direction': (dx, dy),
                'score': score,
                'connected_paths': connected_paths,
                'elevation_change': elevation_change
            })
        
        # 점수순으로 정렬
        available_moves.sort(key=lambda x: x['score'], reverse=True)
        
        return available_moves

    def _is_on_path(self, position):
        """주어진 위치가 길 위인지 확인"""
        x, y = position
        return (self.dataset.road_data[y, x] == 1 or
                self.dataset.forestroad_data[y, x] == 1 or
                self.dataset.climbpath_data[y, x] == 1)


    def step(self, action=None):
        """Execute action and return next state, reward, done"""
        self.timestep += 1
        prev_x, prev_y = self.current_position
        
        if self.following_path:
            # 경로 따라가기 모드에서는 여러 방향 중 하나를 선택
            available_moves = self._get_ranked_path_directions()
            if available_moves:
                # 랜덤 확률로 top 3 중에서 선택
                rand_val = random.random()
                if rand_val < 0.7:  # 70% 확률로 최적 경로
                    dx, dy = available_moves[0]['direction']
                elif rand_val < 0.9 and len(available_moves) > 1:  # 20% 확률로 2번째
                    dx, dy = available_moves[1]['direction']
                elif len(available_moves) > 2:  # 10% 확률로 3번째
                    dx, dy = available_moves[2]['direction']
                else:  # 선택할 수 있는 방향이 부족하면 최적 경로
                    dx, dy = available_moves[0]['direction']
                new_x, new_y = prev_x + dx, prev_y + dy
            else:
                new_x, new_y = prev_x, prev_y
        else:
            dy, dx = self.moves[action]
            new_x, new_y = prev_x + dx, prev_y + dy
            
            if self._is_on_path((new_x, new_y)):
                self.following_path = True
        
        # Boundary and water feature checks
        if (0 <= new_x < self.shape[1] and 
            0 <= new_y < self.shape[0] and 
            not self._is_in_water((new_x, new_y))):
            self.current_position = (new_x, new_y)
        else:
            self.current_position = (prev_x, prev_y)
        
        next_state = self.get_state(self.current_position)
        reward = self._calculate_reward(next_state)
        
        self.visited.add(self.current_position)
        self.path.append(self.current_position)
        
        done = (self.timestep >= self.max_timesteps or 
                self.area_mask[self.current_position[1], self.current_position[0]] == 0)
        
        return next_state, reward, done

    def reset(self, start_pos):
        """Reset environment with given starting position"""
        self.current_position = start_pos
        self.visited.clear()
        self.path.clear()
        self.timestep = 0
        self.following_path = False
        
        # 시작 위치가 이미 길 위인지 확인
        if self._is_on_path(start_pos):
            self.following_path = True
            print("Starting on path! Using path following mode.")
        
        initial_state = self.get_state(self.current_position)
        self.visited.add(self.current_position)
        self.path.append(self.current_position)
        
        return initial_state

    def get_state(self, position):
        """Get state representation for given position"""
        x, y = position
        patch_size = 3
        half_size = patch_size // 2
        
        # Define patch boundaries
        x_start = max(0, x - half_size)
        x_end = min(self.shape[1], x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(self.shape[0], y + half_size + 1)
        
        # Extract patches
        dem_patch = self.dataset.dem[y_start:y_end, x_start:x_end]
        slope_patch = self.dataset.slope[y_start:y_end, x_start:x_end]
        road_patch = self.dataset.road_data[y_start:y_end, x_start:x_end]
        forestroad_patch = self.dataset.forestroad_data[y_start:y_end, x_start:x_end]
        climbpath_patch = self.dataset.climbpath_data[y_start:y_end, x_start:x_end]
        
        state = np.array([
            self.dataset.dem[y, x],                # 현재 고도
            self.dataset.slope[y, x],              # 현재 경사도
            np.mean(dem_patch),                    # 주변 평균 고도
            np.mean(slope_patch),                  # 주변 평균 경사도
            float(np.max(road_patch)),             # 도로 존재 여부
            float(np.max(forestroad_patch)),       # 산림도로 존재 여부
            float(np.max(climbpath_patch)),        # 등산로 존재 여부
            float(self.following_path)             # 길 따라가기 모드 여부
        ])
        
        return state

    def _calculate_reward(self, state):
        """Calculate reward based on current state and environmental features"""
        reward = 0
        x, y = self.current_position
        
        # 물 지역 패널티
        if self._is_in_water((x, y)):
            reward -= 0.5
            return reward
        
        # 이전 위치와의 거리 계산
        if len(self.path) > 1:
            prev_x, prev_y = self.path[-2]
            movement_vector = (x - prev_x, y - prev_y)
            
            # 같은 방향으로 계속 움직이는 것에 대한 보상
            if len(self.path) > 2:
                prev_prev_x, prev_prev_y = self.path[-3]
                prev_vector = (prev_x - prev_prev_x, prev_y - prev_prev_y)
                
                # 방향 다양성 장려
                if movement_vector != prev_vector:
                    reward += 0.08
            
            # 제자리걸음 패널티
            if movement_vector == (0, 0):
                reward -= 0.2
            
            # 왔다갔다 움직임 패널티
            if len(self.path) > 3:
                recent_positions = set(self.path[-4:])
                if len(recent_positions) <= 2:
                    reward -= 0.1
        
        if self.following_path:
            # 길 따라가기 기본 보상
            base_reward = 0.5
            
            # 수로 따라가기 추가 보상
            if self.channels_data[y, x] == 1:
                reward += 0.2
            
            # 물 근처 보상
            if self._is_near_water(x, y, radius=3):
                reward += 0.15
            
            # 진행 방향 유지 보상
            if len(self.path) > 2:
                prev_x, prev_y = self.path[-2]
                prev_prev_x, prev_prev_y = self.path[-3]
                current_direction = (x - prev_x, y - prev_y)
                prev_direction = (prev_x - prev_prev_x, prev_y - prev_prev_y)
                
                if current_direction == prev_direction:
                    reward += 0.1
                    
            # 이미 방문한 길 페널티
            visit_count = sum(1 for pos in self.path if pos == (x, y))
            if visit_count > 1:
                reward -= visit_count * 0.05
                
            reward += base_reward
            
        else:
            # 탐색 모드 보상
            if self._is_on_path((x, y)):
                reward += 1.5  # 길 발견 보상
            else:
                # 거리 기반 보상 수정
                min_dist = self._calculate_min_path_distance(x, y)
                reward += 0.3 / (min_dist + 1)
                
                # 탐색 영역 다양성 보상
                nearby_visits = self._count_nearby_visits(x, y, radius=5)
                reward -= nearby_visits * 0.03
                
                # 경사도 고려
                if self.dataset.slope[y, x] > 30:
                    reward -= 0.05
                
                # 탐색 장려 보상
                if len(self.path) > 1:
                    prev_x, prev_y = self.path[-2]
                    if (x, y) != (prev_x, prev_y):  # 움직임 보상
                        reward += 0.02
        
        # 영역 내 유지 보너스
        if self.area_mask[y, x] == 1:
            reward += 0.03
        
        return reward
        
    def _count_nearby_visits(self, x, y, radius=5):
        """Count number of visits in nearby area"""
        x_start = max(0, x - radius)
        x_end = min(self.shape[1], x + radius + 1)
        y_start = max(0, y - radius)
        y_end = min(self.shape[0], y + radius + 1)
        
        visit_count = 0
        for pos in self.path:
            pos_x, pos_y = pos
            if (x_start <= pos_x < x_end and 
                y_start <= pos_y < y_end):
                visit_count += 1
        
        return visit_count
    
    def _get_path_direction(self):
        """현재 위치에서 길을 따라갈 방향 결정 - 초기 방향 유지"""
        x, y = self.current_position
        available_moves = []
        
        # 초기 진행 방향 결정
        initial_direction = None
        if len(self.path) >= 3:  # 최소 3개 위치로 방향 판단
            first_pos = self.path[0]
            current_pos = self.current_position
            initial_vector = (current_pos[0] - first_pos[0], current_pos[1] - first_pos[1])
            if abs(initial_vector[0]) > abs(initial_vector[1]):  # 주요 이동 방향 결정
                initial_direction = 'horizontal'
            else:
                initial_direction = 'vertical'
        
        # 현재 진행 방향 계산
        current_direction = None
        if len(self.path) > 1:
            prev_x, prev_y = self.path[-2]
            current_direction = (x - prev_x, y - prev_y)
        
        # 8방향 검사
        for dy, dx in self.moves.values():
            new_y, new_x = y + dy, x + dx
            
            # 기본 검증
            if not (0 <= new_x < self.shape[1] and 0 <= new_y < self.shape[0]):
                continue
            if not self._is_on_path((new_x, new_y)):
                continue
            
            # 방문 횟수 확인 (최근 이동만 고려)
            visit_count = sum(1 for pos in self.path[-10:] if pos == (new_x, new_y))
            if visit_count > 1:  # 최근에 재방문한 경우 제외
                continue
            
            # 새로운 위치의 고도
            current_elevation = self.dataset.dem[y, x]
            new_elevation = self.dataset.dem[new_y, new_x]
            elevation_change = abs(new_elevation - current_elevation)
            
            # 급격한 고도 변화가 있는 방향 제외
            if elevation_change > 5:
                continue
            
            # 점수 계산
            score = 0.0
            
            # 1. 초기 방향 유지 보너스
            if initial_direction:
                if initial_direction == 'horizontal':
                    # 수평 방향 선호
                    if abs(dx) > abs(dy):
                        score += 5.0
                else:
                    # 수직 방향 선호
                    if abs(dy) > abs(dx):
                        score += 5.0
            
            # 2. 현재 방향 유지 보너스
            if current_direction:
                if (dx, dy) == current_direction:
                    score += 3.0
                elif abs(dx - current_direction[0]) + abs(dy - current_direction[1]) <= 1:
                    score += 1.5
            
            # 3. 도로 타입 보너스 (일반도로 선호)
            if self.dataset.road_data[new_y, new_x] == 1:
                score += 2.0
            elif self.dataset.forestroad_data[new_y, new_x] == 1:
                score += 1.5
            elif self.dataset.climbpath_data[new_y, new_x] == 1:
                score += 1.0
            
            # 4. 연결성 점수
            connected_paths = sum(1 for next_dy, next_dx in self.moves.values()
                                if (0 <= new_x + next_dx < self.shape[1] and 
                                    0 <= new_y + next_dy < self.shape[0] and 
                                    self._is_on_path((new_x + next_dx, new_y + next_dy))))
            score += connected_paths * 0.5
            
            # 5. 고도 변화 페널티
            score -= elevation_change * 0.2
            
            # 6. 방문 횟수 페널티
            score -= visit_count * 2.0
            
            available_moves.append({
                'direction': (dx, dy),
                'score': score,
                'connected_paths': connected_paths,
                'elevation_change': elevation_change
            })
        
        if not available_moves:
            return None
        
        # 가장 높은 점수의 이동 선택
        best_move = max(available_moves, key=lambda x: x['score'])
        return best_move['direction']

    def _is_path_end(self, current_pos, next_pos):
        """Check if the path should end at the given position"""
        x, y = current_pos
        next_x, next_y = next_pos
        
        # 경계 체크
        if not (0 <= next_x < self.shape[1] and 0 <= next_y < self.shape[0]):
            return True
        
        # 1. 고도 체크
        current_elevation = self.dataset.dem[y, x]
        next_elevation = self.dataset.dem[next_y, next_x]
        elevation_change = abs(next_elevation - current_elevation)
        
        # 2. 연결성 체크 - 다음 위치에서 이어지는 길이 있는지 확인
        connected_paths = 0
        for dy, dx in self.moves.values():
            check_y, check_x = next_y + dy, next_x + dx
            if (0 <= check_x < self.shape[1] and 
                0 <= check_y < self.shape[0] and 
                self._is_on_path((check_x, check_y))):
                connected_paths += 1
        
        # 3. 방문 이력 체크
        if len(self.path) >= 5:  # 최근 5개 위치 확인
            recent_positions = self.path[-5:]
            visit_count = sum(1 for pos in recent_positions if pos == next_pos)
            if visit_count >= 2:  # 최근에 2번 이상 방문했으면 종료
                return True
        
        # 4. 도로 타입에 따른 이동 제한
        current_road_type = None
        if self.dataset.road_data[y, x] == 1:
            current_road_type = "road"
        elif self.dataset.forestroad_data[y, x] == 1:
            current_road_type = "forest"
        elif self.dataset.climbpath_data[y, x] == 1:
            current_road_type = "climb"
        
        next_road_type = None
        if self.dataset.road_data[next_y, next_x] == 1:
            next_road_type = "road"
        elif self.dataset.forestroad_data[next_y, next_x] == 1:
            next_road_type = "forest"
        elif self.dataset.climbpath_data[next_y, next_x] == 1:
            next_road_type = "climb"
        
        # 도로 타입이 급격히 변하는 것을 방지 (예: 일반도로 -> 등산로)
        if current_road_type and next_road_type:
            if ((current_road_type == "road" and next_road_type == "climb") or
                (current_road_type == "climb" and next_road_type == "road")):
                return True
        
        # 종료 조건들
        return any([
            elevation_change > 5,           # 급격한 고도 변화
            connected_paths < 2,            # 낮은 연결성
            not self._is_on_path(next_pos)  # 길을 벗어남
        ])
        
    def _calculate_min_path_distance(self, x, y, search_radius=10):
        """Calculate minimum distance to any path"""
        min_dist = float('inf')
        x_start = max(0, x - search_radius)
        x_end = min(self.shape[1], x + search_radius + 1)
        y_start = max(0, y - search_radius)
        y_end = min(self.shape[0], y + search_radius + 1)
        
        for i in range(y_start, y_end):
            for j in range(x_start, x_end):
                if self._is_on_path((j, i)):
                    dist = np.sqrt((x - j)**2 + (y - i)**2)
                    min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else search_radius

    def _rasterize_data(self, gdf, data_name):
        """Rasterize geodataframe to binary matrix"""
        try:
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            raster = rasterize(
                [(geom, 1) for geom in gdf.geometry],
                out_shape=self.shape,
                transform=self.transform,
                fill=0,
                dtype=np.uint8
            )
            print(f"Rasterized {data_name} successfully")
            return raster
            
        except Exception as e:
            print(f"Error rasterizing {data_name}: {e}")
            return np.zeros(self.shape, dtype=np.uint8)
    
    def load_environmental_data(self):
        """Load and rasterize environmental data"""
        try:
            if rirsv_file:
                rirsv_gdf = gpd.read_file(rirsv_file)
                self.rirsv_data = self._rasterize_data(rirsv_gdf, "reservoir")
            else:
                self.rirsv_data = np.zeros(self.shape, dtype=np.uint8)

            if wkmstrm_file:
                wkmstrm_gdf = gpd.read_file(wkmstrm_file)
                self.wkmstrm_data = self._rasterize_data(wkmstrm_gdf, "stream")
            else:
                self.wkmstrm_data = np.zeros(self.shape, dtype=np.uint8)

            if watershed_file:
                watershed_gdf = gpd.read_file(watershed_file)
                self.watershed_data = self._rasterize_data(watershed_gdf, "watershed")
            else:
                self.watershed_data = np.zeros(self.shape, dtype=np.uint8)

            if channels_file:
                channels_gdf = gpd.read_file(channels_file)
                self.channels_data = self._rasterize_data(channels_gdf, "channels")
            else:
                self.channels_data = np.zeros(self.shape, dtype=np.uint8)
            
        except Exception as e:
            print(f"Error loading environmental data: {e}")
            self.rirsv_data = np.zeros(self.shape, dtype=np.uint8)
            self.wkmstrm_data = np.zeros(self.shape, dtype=np.uint8)
            self.watershed_data = np.zeros(self.shape, dtype=np.uint8)
            self.channels_data = np.zeros(self.shape, dtype=np.uint8)

    def _rasterize_data(self, gdf, data_name):
        try:
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            raster = rasterize(
                [(geom, 1) for geom in gdf.geometry],
                out_shape=self.shape,
                transform=self.transform,
                fill=0,
                dtype=np.uint8
            )
            return raster
        except Exception as e:
            return np.zeros(self.shape, dtype=np.uint8)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)