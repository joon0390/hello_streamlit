import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import rasterize
from pyproj import CRS, Transformer

class CustomGeoDataset:
    def __init__(self, dem_file, road_file, forestroad_file, climbpath_file, 
                 rirsv_file=None, wkmstrm_file=None, watershed_file=None, channels_file=None):
        
        with rasterio.open(dem_file) as src:
            #print(f"Original DEM CRS: {src.crs}")
            target_crs = CRS.from_epsg(32652)
            if src.crs != target_crs:
                #print(f"Converting DEM from {src.crs} to EPSG:32652")
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
                # print(f"Reprojected DEM shape: {self.dem.shape}")
                # print(f"Reprojected DEM transform: {self.transform}")
                # print(f"Reprojected DEM CRS: {self.crs}")

            else:
                # 좌표계 변환이 필요하지 않은 경우
                #print("DEM is already in EPSG:32652")
                self.dem = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
                self.shape = src.shape

        # 도로 데이터 로드 및 벡터화
        self.road_data = self._vectorize_roads(road_file, "도로")
        self.forestroad_data = self._vectorize_roads(forestroad_file, "산림도로")
        self.climbpath_data = self._vectorize_roads(climbpath_file, "등산로")

        # 추가 환경 데이터 로드
        self.rirsv_data = self._vectorize_roads(rirsv_file, "저수지") if rirsv_file else np.zeros(self.shape)
        self.wkmstrm_data = self._vectorize_roads(wkmstrm_file, "하천") if wkmstrm_file else np.zeros(self.shape)
        self.watershed_data = self._vectorize_roads(watershed_file, "유역") if watershed_file else np.zeros(self.shape)
        self.channels_data = self._vectorize_roads(channels_file, "수로") if channels_file else np.zeros(self.shape)

        # 경사도 계산
        self.slope = self._calculate_slope()

    def _reproject_dem(self, src, target_crs):
        """DEM 데이터를 EPSG:32652로 변환"""
        transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)
        
        # 새로운 transform을 적용한 좌표로 변환
        dem_data = src.read(1)
        reprojected_data = np.empty_like(dem_data)

        # 각 픽셀의 좌표를 변환
        for i in range(dem_data.shape[0]):
            for j in range(dem_data.shape[1]):
                x, y = rasterio.transform.xy(src.transform, i, j)
                x_new, y_new = transformer.transform(x, y)
                reprojected_data[i, j] = dem_data[i, j]  # 보간을 추가하여 필요시 조정 가능

        # 좌표 변환된 데이터와 원본과 동일한 transform 반환
        return reprojected_data, src.transform

    def _vectorize_roads(self, road_file, road_type="road"):
        """벡터 데이터를 래스터화하여 이진 행렬로 변환"""
        try:
            roads = gpd.read_file(road_file)
            
            # CRS 변환이 필요한 경우 처리
            #print(f"{road_type} original CRS: {roads.crs}")
            if roads.crs != self.crs:
                roads = roads.to_crs(self.crs)
                #print(f"{road_type} coordinate system transformed to {self.crs}")
            
            road_raster = rasterize(
                [(geom, 1) for geom in roads.geometry],
                out_shape=self.shape,
                transform=self.transform,
                fill=0,
                dtype=np.uint8
            )
            #print(f"{road_type} rasterized successfully with shape: {road_raster.shape}")
            return road_raster
            
        except Exception as e:
            #print(f"{road_type} vectorization error: {e}")
            return np.zeros(self.shape, dtype=np.uint8)

    def _calculate_slope(self):
        """Calculate slope from DEM"""
        dy, dx = np.gradient(self.dem)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        print("Slope calculated successfully")
        return slope

    def __getitem__(self, idx):
        """Get state representation for a given position"""
        x, y = idx
        patch_size = 3  # 3x3 neighborhood
        half_size = patch_size // 2
        
        # 패치 내 경계를 넘지 않도록 설정
        x_start = max(0, x - half_size)
        x_end = min(self.shape[1], x + half_size + 1)
        y_start = max(0, y - half_size)
        y_end = min(self.shape[0], y + half_size + 1)
        
        # 주변의 DEM 및 도로 데이터 패치 추출
        dem_patch = self.dem[y_start:y_end, x_start:x_end]
        slope_patch = self.slope[y_start:y_end, x_start:x_end]
        road_patch = self.road_data[y_start:y_end, x_start:x_end]
        forestroad_patch = self.forestroad_data[y_start:y_end, x_start:x_end]
        climbpath_patch = self.climbpath_data[y_start:y_end, x_start:x_end]
        
        # 8개 특징 계산
        state = np.array([
            self.dem[y, x],  # 현재 고도
            self.slope[y, x],  # 현재 경사도
            np.mean(dem_patch),  # 주변 평균 고도
            np.mean(slope_patch),  # 주변 평균 경사도
            np.max(road_patch),  # 도로 여부
            np.max(forestroad_patch),  # 산림도로 여부
            np.max(climbpath_patch),  # 등산로 여부
            np.std(dem_patch)  # 지형 거칠기
        ])
        
        return state