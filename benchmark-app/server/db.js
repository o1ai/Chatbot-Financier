const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgres://localhost/benchmarkdb'
});

module.exports = {
  query: (text, params) => pool.query(text, params)
};
