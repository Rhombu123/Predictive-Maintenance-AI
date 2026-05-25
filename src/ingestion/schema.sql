-- src/ingestion/schema.sql
CREATE TABLE sensor_readings (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(50),
    timestamp TIMESTAMPTZ NOT NULL,
    vibration FLOAT,
    temperature FLOAT,
    pressure FLOAT,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equipment_time ON sensor_readings(equipment_id, timestamp DESC);