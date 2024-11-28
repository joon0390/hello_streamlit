import rasterio
from pyproj import CRS, Transformer

transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)

def pixel_to_gis_coordinates(x, y, transform):
    lon, lat = rasterio.transform.xy(transform, y, x)
    return [lon, lat]

def utm_to_latlon(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon

def pixel_to_latlon(x, y, transform):
    utm_coords = transform * (x, y)
    utm_x, utm_y = float(utm_coords[0]), float(utm_coords[1])
    result = transformer.transform(utm_x, utm_y)
    return [float(result[0]), float(result[1])]

def latlon_to_pixel(lat, lon, transform, crs):
    utm_transformer = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
    utm_x, utm_y = utm_transformer.transform(lon, lat)
    row, col = ~transform * (utm_x, utm_y)
    return int(col), int(row)