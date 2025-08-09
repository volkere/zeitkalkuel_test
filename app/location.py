
from __future__ import annotations
from typing import Optional, Dict
from PIL import Image, ExifTags

def extract_exif_gps(image_path: str) -> Optional[Dict[str, float]]:
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        exif_data = { ExifTags.TAGS.get(k,k): v for k,v in exif.items() }
        gps_info = exif_data.get('GPSInfo')
        if not gps_info:
            return None
        gps_parsed = { ExifTags.GPSTAGS.get(t, t): gps_info[t] for t in gps_info }
        from .utils import dms_to_dd
        lat = dms_to_dd(gps_parsed.get('GPSLatitude'), gps_parsed.get('GPSLatitudeRef'))
        lon = dms_to_dd(gps_parsed.get('GPSLongitude'), gps_parsed.get('GPSLongitudeRef'))
        if lat is None or lon is None:
            return None
        return {'lat': lat, 'lon': lon}
    except Exception:
        return None

def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="photo_metadata_app")
        location = geolocator.reverse((lat, lon), timeout=10, exactly_one=True, language="en")
        if location:
            return location.address
        return None
    except Exception:
        return None
