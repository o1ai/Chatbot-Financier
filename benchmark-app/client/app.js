const { useState, useEffect } = React;

function App() {
  const [benchmarks, setBenchmarks] = useState([]);

  useEffect(() => {
    fetch('/api/benchmarks')
      .then(res => res.json())
      .then(setBenchmarks)
      .catch(err => console.error(err));
  }, []);

  return (
    <div>
      <h1>AI Tools Benchmark (2024 - Present)</h1>
      <table border="1" cellPadding="5">
        <thead>
          <tr>
            <th>Year</th>
            <th>Tool</th>
            <th>Category</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {benchmarks.map(row => (
            <tr key={row.id}>
              <td>{row.year}</td>
              <td>{row.tool}</td>
              <td>{row.category}</td>
              <td>{row.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

ReactDOM.render(<App />, document.getElementById('root'));
