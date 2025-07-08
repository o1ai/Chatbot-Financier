CREATE TABLE IF NOT EXISTS benchmarks (
  id SERIAL PRIMARY KEY,
  year INT NOT NULL,
  tool TEXT NOT NULL,
  category TEXT NOT NULL,
  score NUMERIC NOT NULL
);

INSERT INTO benchmarks (year, tool, category, score) VALUES
(2024, 'ChatGPT', 'development', 8.5),
(2024, 'Claude', 'database', 7.2),
(2024, 'Gemini', 'problem_solving', 8.8),
(2025, 'ChatGPT', 'database', 7.9);
