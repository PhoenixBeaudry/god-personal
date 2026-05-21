-- migrate:up
CREATE TABLE IF NOT EXISTS pvp_pair_results (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey_a TEXT NOT NULL,
    hotkey_b TEXT NOT NULL,
    environment_name TEXT NOT NULL,
    model_a_wins INT NOT NULL DEFAULT 0,
    model_b_wins INT NOT NULL DEFAULT 0,
    draws INT NOT NULL DEFAULT 0,
    total_games INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey_a, hotkey_b, environment_name)
);

-- migrate:down
DROP TABLE IF EXISTS pvp_pair_results;
