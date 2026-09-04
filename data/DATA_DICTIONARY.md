# Data Dictionary

## traffic.csv
| Column | Description |
|---|---|
| timestamp | Date and time (hourly) |
| zone_id | City zone (Z1-Z6) |
| vehicle_count | Number of vehicles observed |
| average_speed | Average vehicle speed (km/h) |
| congestion_level | Low / Medium / High |
| road_type | Highway / Residential / Commercial |
| accident_count | Number of accidents that hour |
| is_event | 1 if a city event overlapped this hour/zone |

## weather.csv
| Column | Description |
|---|---|
| timestamp | Date/time (hourly) |
| temperature | °C |
| humidity | % |
| rainfall | mm |
| wind_speed | km/h |
| pressure | hPa |
| visibility | km |

## air_quality.csv
| Column | Description |
|---|---|
| timestamp, zone_id | Grain |
| AQI | Air Quality Index |
| PM2_5, PM10 | Particulate matter (µg/m³) |
| NO2, CO, SO2, O3 | Pollutant concentrations |

## energy.csv
| Column | Description |
|---|---|
| timestamp, zone_id | Grain |
| energy_consumption | kWh |
| peak_demand | kWh |

## transport.csv
| Column | Description |
|---|---|
| timestamp | Every 3 hours |
| route_id | Bus/transit route |
| passenger_count | Passengers carried |
| delay_minutes | Average delay |
| vehicle_count | Vehicles on the route |

## complaints.csv
| Column | Description |
|---|---|
| complaint_id | Unique ID |
| timestamp, zone_id | Grain |
| complaint_text | Free-text complaint |
| category | Traffic/Road/Water/Electricity/Pollution/Transport/Other |
| status | Open/Closed |
| priority | Low/Medium/High |

## events.csv
| Column | Description |
|---|---|
| event_id, event_name | Event identifiers |
| zone_id | Location |
| start_time, end_time | Duration |
| expected_crowd | Estimated attendance |

## data/feature_engineered/master.csv
Traffic joined with weather, air quality and energy at the
(timestamp, zone_id) grain, plus engineered columns: `hour`, `day`,
`month`, `day_of_week`, `is_weekend`, `is_peak`, and lag/rolling
features on `vehicle_count` (`vehicle_count_lag_1`, `_lag_24`,
`_rolling_mean_24`, `_rolling_std_24`).

> Note: all data here is **synthetically generated** (see
> `src/data/generate_data.py`) so the project runs end-to-end without
> external data access. Swap in real collectors/APIs for production use.
