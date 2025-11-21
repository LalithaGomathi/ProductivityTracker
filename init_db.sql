
CREATE TABLE IF NOT EXISTS agents (
  id SERIAL PRIMARY KEY,
  agent_id TEXT UNIQUE NOT NULL,
  display_name TEXT
);
