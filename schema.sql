CREATE DATABASE IF NOT EXISTS sentinel_ai
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE sentinel_ai;

-- Base table for a fresh installation. The migration script adds canonical
-- event, classifier, alert, and investigation columns to this table.
CREATE TABLE IF NOT EXISTS live_security_events (
    event_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    detected_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    raw_event JSON NOT NULL,
    anomaly_score DOUBLE NOT NULL,
    status VARCHAR(20) NOT NULL,
    prediction TINYINT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
