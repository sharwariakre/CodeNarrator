import json
from pathlib import Path
from typing import Dict

from app.services.analysis_snapshot_service import _compute_dependency_graph_summary


def generate_html_report(final_state: Dict, interpretation: Dict | None, output_path: Path) -> Path:
    payload = _build_report_payload(final_state, interpretation)
    html = _render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_report_payload(final_state: Dict, interpretation: Dict | None) -> Dict:
    summary = final_state.get("current_summary", {})
    graph_summary = _compute_dependency_graph_summary(final_state)
    inspected_facts = final_state.get("inspected_facts", [])
    explored_files = final_state.get("explored_files", [])

    fact_map = {fact.get("file_path"): fact for fact in inspected_facts}
    cluster_lookup: Dict[str, str] = {}
    for cluster in graph_summary.get("clusters", []):
        label = cluster.get("cluster", "unclustered")
        for file_path in cluster.get("files", []):
            cluster_lookup[file_path] = label

    internal_edges = graph_summary.get("internal_edges", [])
    explored_set = set(explored_files)
    visible_edges = [
        {"source": edge["from"], "target": edge["to"]}
        for edge in internal_edges
        if edge.get("from") in explored_set and edge.get("to") in explored_set
    ]

    incoming_count: Dict[str, int] = {file_path: 0 for file_path in explored_files}
    for edge in visible_edges:
        target = edge.get("to")
        if target in incoming_count:
            incoming_count[target] += 1

    nodes = []
    for file_path in explored_files:
        fact = fact_map.get(file_path, {})
        nodes.append(
            {
                "id": file_path,
                "cluster": cluster_lookup.get(file_path, "unclustered"),
                "role_hint": fact.get("role_hint", "unknown"),
                "language": fact.get("language", "unknown"),
                "imports_count": fact.get("imports_found", 0),
                "imported_modules": fact.get("imported_modules", []),
                "line_count_bucket": fact.get("line_count_bucket", "unknown"),
                "in_degree": incoming_count.get(file_path, 0),
            }
        )
    node_ids = {node["id"] for node in nodes}
    visible_edges = [
        edge
        for edge in visible_edges
        if edge.get("source") in node_ids and edge.get("target") in node_ids
    ]

    interpretation = interpretation or {}
    main_components = interpretation.get("main_components", [])
    key_dependencies = interpretation.get("key_dependencies", [])
    pattern = interpretation.get("architecture_pattern", "Not available")
    summary_text = interpretation.get(
        "summary_for_new_developer",
        "AI interpretation not available for this run.",
    )

    return {
        "repo": {
            "name": summary.get("repo", "unknown"),
            "file_count": summary.get("file_count", 0),
            "languages": summary.get("languages", []),
            "confidence": final_state.get("confidence", 0.0),
        },
        "graph": {
            "nodes": nodes,
            "links": visible_edges,
            "clusters": graph_summary.get("clusters", []),
        },
        "inspected_facts": inspected_facts,
        "ai": {
            "architecture_pattern": pattern,
            "main_components": main_components,
            "key_dependencies": key_dependencies,
            "summary_for_new_developer": summary_text,
        },
    }


def _render_html(payload: Dict) -> str:
    data_json = json.dumps(payload).replace("</script>", "<\\/script>")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CodeNarrator Architecture Report</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --line: #d1d5db;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 35%);
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 3px 10px rgba(0,0,0,0.04);
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .meta .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfdff;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .value {{
      font-size: 18px;
      font-weight: 700;
      margin-top: 4px;
    }}
    #graph {{
      width: 100%;
      height: 560px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }}
    .legend {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .component, .dep {{
      border-left: 4px solid var(--accent);
      padding: 10px 12px;
      margin-bottom: 10px;
      background: #f9fffe;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
      font-weight: 600;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: #111827;
      color: #fff;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      max-width: 420px;
      opacity: 0;
      transition: opacity 120ms ease;
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="panel">
      <h1 id="repo-title"></h1>
      <p><strong>Architecture Pattern:</strong> <span id="arch-pattern"></span></p>
      <p id="new-dev-summary"></p>
      <div class="meta">
        <div class="card"><div class="label">File Count</div><div class="value" id="meta-files"></div></div>
        <div class="card"><div class="label">Languages</div><div class="value" id="meta-langs"></div></div>
        <div class="card"><div class="label">Confidence</div><div class="value" id="meta-confidence"></div></div>
      </div>
    </section>

    <section class="panel">
      <h2>Dependency Graph</h2>
      <svg id="graph" width="100%" height="560"></svg>
      <div class="legend">Node size = incoming internal dependencies. Color = cluster.</div>
    </section>

    <section class="panel">
      <h2>Main Components</h2>
      <div id="components"></div>
    </section>

    <section class="panel">
      <h2>Key Dependencies</h2>
      <div id="key-dependencies"></div>
    </section>

    <section class="panel">
      <h2>Explored Files</h2>
      <table>
        <thead>
          <tr>
            <th>File Path</th>
            <th>Language</th>
            <th>Role</th>
            <th>Imports Count</th>
            <th>Line Count Bucket</th>
          </tr>
        </thead>
        <tbody id="files-table"></tbody>
      </table>
    </section>
  </div>
  <div class="tooltip" id="tooltip"></div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
  <script>
    window.REPORT_DATA = {data_json};

    const data = window.REPORT_DATA;
    document.getElementById("repo-title").textContent = data.repo.name;
    document.getElementById("arch-pattern").textContent = data.ai.architecture_pattern || "Not available";
    document.getElementById("new-dev-summary").textContent = data.ai.summary_for_new_developer || "Not available";
    document.getElementById("meta-files").textContent = data.repo.file_count;
    document.getElementById("meta-langs").textContent = (data.repo.languages || []).join(", ");
    document.getElementById("meta-confidence").textContent = data.repo.confidence;

    const componentsRoot = document.getElementById("components");
    if ((data.ai.main_components || []).length === 0) {{
      componentsRoot.innerHTML = "<p>No AI component interpretation available.</p>";
    }} else {{
      data.ai.main_components.forEach(c => {{
        const el = document.createElement("div");
        el.className = "component";
        el.innerHTML = `<strong>${{c.name}}</strong><p>${{c.description}}</p><p><em>${{(c.files || []).join(", ")}}</em></p>`;
        componentsRoot.appendChild(el);
      }});
    }}

    const depsRoot = document.getElementById("key-dependencies");
    if ((data.ai.key_dependencies || []).length === 0) {{
      depsRoot.innerHTML = "<p>No AI dependency interpretation available.</p>";
    }} else {{
      data.ai.key_dependencies.forEach(dep => {{
        const el = document.createElement("div");
        el.className = "dep";
        el.innerHTML = `<strong>${{dep.from}} → ${{dep.to}}</strong><p>${{dep.reason}}</p>`;
        depsRoot.appendChild(el);
      }});
    }}

    const tableRoot = document.getElementById("files-table");
    (data.inspected_facts || []).forEach(f => {{
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${{f.file_path}}</td>
        <td>${{f.language}}</td>
        <td>${{f.role_hint}}</td>
        <td>${{f.imports_found}}</td>
        <td>${{f.line_count_bucket}}</td>
      `;
      tableRoot.appendChild(tr);
    }});

    const nodes = (data.graph.nodes || []).map(d => ({{...d}}));
    const links = (data.graph.links || []).map(d => ({{...d}}));

    const svg = d3.select("#graph");
    const bbox = svg.node().getBoundingClientRect();
    const width = bbox.width || 960;
    const height = bbox.height || 560;
    const tooltip = d3.select("#tooltip");

    const clusterDomain = [...new Set(nodes.map(n => n.cluster))];
    const color = d3.scaleOrdinal(clusterDomain, d3.schemeTableau10);

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(d => 8 + (d.in_degree || 0) * 2));

    svg.append("g")
      .attr("stroke", "#9ca3af")
      .attr("stroke-opacity", 0.7)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.4);

    const node = svg.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.2)
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", d => 6 + Math.min(14, (d.in_degree || 0) * 2))
      .attr("fill", d => color(d.cluster))
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("mousemove", (event, d) => {{
        tooltip.style("opacity", 1)
          .style("left", (event.pageX + 12) + "px")
          .style("top", (event.pageY + 12) + "px")
          .html(`
            <div><strong>${{d.id}}</strong></div>
            <div>role: ${{d.role_hint}}</div>
            <div>imports_count: ${{d.imports_count}}</div>
            <div>imports: ${{(d.imported_modules || []).join(", ") || "none"}}</div>
          `);
      }})
      .on("mouseleave", () => tooltip.style("opacity", 0));

    simulation.on("tick", () => {{
      svg.selectAll("line")
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
    }});

    function dragstarted(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }}
    function dragged(event, d) {{
      d.fx = event.x;
      d.fy = event.y;
    }}
    function dragended(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }}
  </script>
</body>
</html>"""
