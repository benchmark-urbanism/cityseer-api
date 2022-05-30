def from_latlon(
    latitude: float, longitude: float, force_zone_number: int | None = ..., force_zone_letter: str | None = ...
) -> tuple[float, float, int, str]: ...
def to_latlon(
    easting: float,
    northing: float,
    zone_number: int,
    zone_letter: str | None = ...,
    northern: bool | None = ...,
    strict: bool = ...,
) -> tuple[float, float]: ...
