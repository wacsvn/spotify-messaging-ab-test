-- Target Platform: Google BigQuery
-- Purpose: Collapse daily rows into user-level weekly averages, then summarize by variant.
-- Unit notes:
--   * "weekly" = 28-day total / 4, per ENROLLED user (churned users contribute zeros for missing days).
--   * Crash rate is per user-day row (one app_session_summary row per user per day), a proxy for
--     per-session crash rate. Value is already a percentage (0.206 means 0.206%).
WITH user_level_totals AS (
  SELECT
    user_id,
    variant,
    MAX(is_complier) AS is_complier,

    SUM(total_minutes)    / 4.0 AS weekly_streaming_minutes,
    SUM(autoplay_minutes) / 4.0 AS weekly_autoplay_minutes,
    SUM(search_minutes)   / 4.0 AS weekly_search_minutes,
    SUM(search_sessions)  / 4.0 AS weekly_search_sessions,

    SUM(crashes)            AS total_user_crashes,
    COUNT(experiment_day)   AS total_user_active_days,

    -- W4 retention: any active presence on days 21-28
    MAX(CASE WHEN experiment_day >= 21 THEN w4_active_day ELSE 0 END) AS w4_retained_flag
  FROM `Spotify_AB_Simulated_Traffic_Data.raw_traffic_log`
  GROUP BY user_id, variant
)
SELECT
  variant,
  COUNT(DISTINCT user_id)                                          AS group_sample_size,
  ROUND(SUM(is_complier) / COUNT(DISTINCT user_id), 4)             AS compliance_rate,

  -- Primary OEC
  ROUND(AVG(weekly_streaming_minutes), 2)                          AS mean_weekly_streaming_mins,
  ROUND(STDDEV_SAMP(weekly_streaming_minutes), 2)                  AS std_weekly_streaming_mins,

  -- Secondary / cannibalization
  ROUND(AVG(weekly_autoplay_minutes), 2)                           AS mean_weekly_autoplay_mins,
  ROUND(AVG(weekly_search_minutes), 2)                             AS mean_weekly_search_mins,
  ROUND(AVG(weekly_search_sessions), 2)                            AS mean_weekly_search_sessions,

  -- Guardrails
  ROUND(100 * SUM(total_user_crashes) / SUM(total_user_active_days), 3) AS crash_rate_per_user_day_pct,
  ROUND(100 * SUM(w4_retained_flag) / COUNT(DISTINCT user_id), 2)       AS w4_retention_rate_pct
FROM user_level_totals
GROUP BY variant
ORDER BY variant DESC;