from pyproj import Transformer


def convert_to_webmercator(lon_min, lon_max, lat_min, lat_max):
    # Define the transformer for WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Convert the corners of the bounding box
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    return (x_min, x_max, y_min, y_max)


# Seoul range
seoul_lon_min, seoul_lon_max = 126.8255, 127.1516
seoul_lat_min, seoul_lat_max = 37.4738, 37.6202
seoul_webmercator = convert_to_webmercator(seoul_lon_min, seoul_lon_max, seoul_lat_min, seoul_lat_max)
print(f"Seoul Web Mercator range: {seoul_webmercator}")

# Incheon range
incheon_lon_min, incheon_lon_max = 126.5987, 126.8068
incheon_lat_min, incheon_lat_max = 37.3648, 37.5910
incheon_webmercator = convert_to_webmercator(incheon_lon_min, incheon_lon_max, incheon_lat_min, incheon_lat_max)
print(f"Incheon Web Mercator range: {incheon_webmercator}")
