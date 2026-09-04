-- CityPulse AI database schema
-- Written for SQLite (used by src/data/load_to_sql.py) but kept close to
-- standard SQL so it can be ported to SQL Server with minor type changes
-- (e.g. VARCHAR sizes, DATETIME2, IDENTITY instead of AUTOINCREMENT).

CREATE TABLE IF NOT EXISTS locations (
    zone_id     VARCHAR(10) PRIMARY KEY,
    zone_name   VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS traffic (
    id                INTEGER PRIMARY KEY,
    timestamp         DATETIME NOT NULL,
    zone_id           VARCHAR(10) NOT NULL,
    vehicle_count     INTEGER,
    average_speed     REAL,
    congestion_level  VARCHAR(10),
    road_type         VARCHAR(20),
    accident_count    INTEGER DEFAULT 0,
    is_event          INTEGER DEFAULT 0,
    FOREIGN KEY (zone_id) REFERENCES locations(zone_id)
);

CREATE TABLE IF NOT EXISTS weather (
    id           INTEGER PRIMARY KEY,
    timestamp    DATETIME NOT NULL UNIQUE,
    temperature  REAL,
    humidity     REAL,
    rainfall     REAL,
    wind_speed   REAL,
    pressure     REAL,
    visibility   REAL
);

CREATE TABLE IF NOT EXISTS air_quality (
    id         INTEGER PRIMARY KEY,
    timestamp  DATETIME NOT NULL,
    zone_id    VARCHAR(10) NOT NULL,
    AQI        REAL,
    PM2_5      REAL,
    PM10       REAL,
    NO2        REAL,
    CO         REAL,
    SO2        REAL,
    O3         REAL,
    FOREIGN KEY (zone_id) REFERENCES locations(zone_id)
);

CREATE TABLE IF NOT EXISTS energy (
    id                  INTEGER PRIMARY KEY,
    timestamp           DATETIME NOT NULL,
    zone_id             VARCHAR(10) NOT NULL,
    energy_consumption  REAL,
    peak_demand         REAL,
    FOREIGN KEY (zone_id) REFERENCES locations(zone_id)
);

CREATE TABLE IF NOT EXISTS transport (
    id                INTEGER PRIMARY KEY,
    timestamp         DATETIME NOT NULL,
    route_id          VARCHAR(10) NOT NULL,
    passenger_count   INTEGER,
    delay_minutes     REAL,
    vehicle_count     INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    event_id        VARCHAR(10) PRIMARY KEY,
    event_name      VARCHAR(100),
    zone_id         VARCHAR(10),
    start_time      DATETIME,
    end_time        DATETIME,
    expected_crowd  INTEGER,
    FOREIGN KEY (zone_id) REFERENCES locations(zone_id)
);

CREATE TABLE IF NOT EXISTS complaints (
    complaint_id     VARCHAR(10) PRIMARY KEY,
    timestamp        DATETIME,
    zone_id          VARCHAR(10),
    complaint_text   TEXT,
    category         VARCHAR(30),
    status           VARCHAR(10) DEFAULT 'Open',
    priority         VARCHAR(10),
    FOREIGN KEY (zone_id) REFERENCES locations(zone_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY,
    zone_id      VARCHAR(10),
    task         VARCHAR(30),
    predicted    VARCHAR(50),
    probability  REAL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_results (
    id             INTEGER PRIMARY KEY,
    model_name     VARCHAR(50),
    task           VARCHAR(30),
    metric_name    VARCHAR(20),
    metric_value   REAL,
    selected       INTEGER DEFAULT 0,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Example view: zone-wise average traffic
CREATE VIEW IF NOT EXISTS view_zone_avg_traffic AS
SELECT zone_id, AVG(vehicle_count) AS avg_traffic
FROM traffic
GROUP BY zone_id
ORDER BY avg_traffic DESC;
