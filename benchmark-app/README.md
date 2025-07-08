# AI Tools Benchmark Web App

This sample application demonstrates a simple benchmarking dashboard for various AI tools using a Node.js backend with a Postgres database and a React frontend.

## Setup

1. **Install dependencies**
   ```bash
   cd benchmark-app/server
   npm install
   ```

2. **Create the database**
   ```bash
   createdb benchmarkdb
   psql benchmarkdb -f ../sql/init.sql
   ```

3. **Run the server**
   ```bash
   node index.js
   ```

The server runs on `http://localhost:3001` and serves the React client.

Open `http://localhost:3001` in your browser to view the benchmark table.
