async function loadCharts() {
  let j = null;
  try {
    const res = await fetch('/dashboard/chart-data');
    if (res.ok) j = await res.json();
  } catch (e) {}

  const history_dates = j?.history_dates || ["2025-01-01","2025-02-15","2025-03-20","2025-04-10","2025-05-01"];
  const history_scores = j?.history_scores || [2.5, 3.0, 2.8, 3.2, 3.1];
  const confidence_labels = j?.confidence_labels || ["Scan 1","Scan 2","Scan 3","Scan 4","Scan 5"];
  const confidence_values = j?.confidence_values || [70, 85, 60, 90, 75];
  const radar_labels = j?.radar_labels || ["Upper Left","Upper Right","Lower Left","Lower Right","Central"];
  const radar_values = j?.radar_values || [3.0, 2.8, 3.2, 2.9, 3.1];
  const benchmark_labels = j?.benchmark_labels || ["You","Population Avg"];
  const benchmark_values = j?.benchmark_values || [3.2, 2.5];
  const result_labels = j?.result_labels || ["Benign","Malignant","Inconclusive"];
  const result_counts = j?.result_counts || [60, 30, 10];
  const time_bins = j?.time_bins || ["0–30 days","31–60 days","61–90 days","91–120 days"];
  const time_counts = j?.time_counts || [2, 3, 1, 1];

  new Chart(document.getElementById("lineChartHistory"), {
    type: "line",
    data: {
      labels: history_dates,
      datasets: [{
        label: "Health Score",
        data: history_scores,
        borderColor: "rgba(54,162,235,0.8)",
        backgroundColor: "rgba(54,162,235,0.4)",
        tension: 0.2,
        fill: true
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.2)" } },
        y: { beginAtZero:true, grid: { color: "rgba(255,255,255,0.2)" } }
      }
    }
  });

  new Chart(document.getElementById("barChartConfidence"), {
    type: "bar",
    data: {
      labels: confidence_labels,
      datasets:[{
        label: "Confidence (%)",
        data: confidence_values,
        backgroundColor: "rgba(255,159,64,0.6)",
        borderColor:     "rgba(255,159,64,1)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero:true, max:100, grid:{color:"rgba(255,255,255,0.2)"} },
        x: { grid:{color:"rgba(255,255,255,0.2)"} }
      }
    }
  });

  new Chart(document.getElementById("radarChartHealthTrend"), {
    type: "radar",
    data: {
      labels: radar_labels,
      datasets:[{
        label: "Score",
        data: radar_values,
        backgroundColor: "rgba(153,102,255,0.4)",
        borderColor:     "rgba(153,102,255,1)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scale: {
        ticks: { beginAtZero:true },
        grid:  { color:"rgba(255,255,255,0.2)" }
      }
    }
  });

  new Chart(document.getElementById("barChartBenchmark"), {
    type: "bar",
    data: {
      labels: benchmark_labels,
      datasets:[{
        label: "Risk Score",
        data: benchmark_values,
        backgroundColor: ["rgba(75,192,192,0.6)","rgba(255,99,132,0.6)"],
        borderColor:     ["rgba(75,192,192,1)","rgba(255,99,132,1)"],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero:true, grid:{color:"rgba(255,255,255,0.2)"} },
        x: { grid:{color:"rgba(255,255,255,0.2)"} }
      }
    }
  });

  new Chart(document.getElementById("doughnutChartResults"), {
    type: "doughnut",
    data: {
      labels: result_labels,
      datasets:[{
        data: result_counts,
        backgroundColor:[
          "rgba(54,162,235,0.6)",
          "rgba(255,99,132,0.6)",
          "rgba(255,205,86,0.6)"
        ]
      }]
    },
    options: { responsive:true }
  });

  new Chart(document.getElementById("histogramChartTimeBetweenScans"), {
    type: "bar",
    data: {
      labels: time_bins,
      datasets:[{
        label: "Count of Scans",
        data: time_counts,
        backgroundColor: "rgba(255,159,64,0.6)",
        borderColor:     "rgba(255,159,64,1)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero:true, grid:{color:"rgba(255,255,255,0.2)"} },
        x: { grid:{color:"rgba(255,255,255,0.2)"} }
      }
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadCharts);
} else {
  loadCharts();
}
