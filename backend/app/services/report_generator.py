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
    fact_map = {fact.get("file_path"): fact for fact in inspected_facts}
    node_file_paths = [fact.get("file_path") for fact in inspected_facts if fact.get("file_path")]
    cluster_lookup: Dict[str, str] = {}
    for cluster in graph_summary.get("clusters", []):
        label = cluster.get("cluster", "unclustered")
        for file_path in cluster.get("files", []):
            cluster_lookup[file_path] = label

    internal_edges = graph_summary.get("internal_edges", [])
    explored_set = set(node_file_paths)
    # Include edges where the source was explored, even if the target was not.
    visible_edges = [
        {"source": edge["from"], "target": edge["to"]}
        for edge in internal_edges
        if edge.get("from") in explored_set
    ]

    # Targets that were never explored become phantom nodes.
    phantom_ids = {
        edge["target"] for edge in visible_edges
        if edge["target"] not in explored_set
    }

    all_node_ids = list(node_file_paths) + list(phantom_ids)
    incoming_count: Dict[str, int] = {file_path: 0 for file_path in all_node_ids}
    for edge in visible_edges:
        target = edge.get("target")
        if target in incoming_count:
            incoming_count[target] += 1

    nodes = []
    for file_path in node_file_paths:
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
    for file_path in phantom_ids:
        nodes.append(
            {
                "id": file_path,
                "cluster": "unvisited",
                "role_hint": "unknown",
                "language": "unknown",
                "imports_count": 0,
                "imported_modules": [],
                "line_count_bucket": "unknown",
                "in_degree": incoming_count.get(file_path, 0),
            }
        )
    node_ids = {node["id"] for node in nodes}

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
      height: 680px;
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
      <svg id="graph" width="100%" height="680"></svg>
      <div class="legend">Node size = incoming internal dependencies. Color = cluster. Grey dashed = imported but not explored.</div>
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
    const height = bbox.height || 680;
    const tooltip = d3.select("#tooltip");

    const clusterDomain = [...new Set(nodes.map(n => n.cluster).filter(c => c !== "unvisited"))];
    const _color = d3.scaleOrdinal(clusterDomain, d3.schemeTableau10);
    const color = c => c === "unvisited" ? "#c4c8d0" : _color(c);

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide().radius(d => 8 + (d.in_degree || 0) * 2));

    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#9ca3af");

    svg.append("g")
      .attr("stroke", "#9ca3af")
      .attr("stroke-opacity", 0.7)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.4)
      .attr("marker-end", "url(#arrowhead)");

    const node = svg.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.2)
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", d => 10 + Math.min(14, (d.in_degree || 0) * 2))
      .attr("fill", d => color(d.cluster))
      .attr("stroke-dasharray", d => d.cluster === "unvisited" ? "4,2" : null)
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

    const labels = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .text(d => d.id.split("/").pop())
      .attr("font-size", "11px")
      .attr("fill", "#374151")
      .attr("dx", 12)
      .attr("dy", 4);

    simulation.on("tick", () => {{
      nodes.forEach(d => {{
        const r = 10 + Math.min(14, (d.in_degree || 0) * 2);
        d.x = Math.max(r, Math.min(width - r, d.x));
        d.y = Math.max(r, Math.min(height - r, d.y));
      }});

      svg.selectAll("line")
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      labels
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    }});

    function dragstarted(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }}
    function dragged(event, d) {{
      const r = 10 + Math.min(14, (d.in_degree || 0) * 2);
      d.fx = Math.max(r, Math.min(width - r, event.x));
      d.fy = Math.max(r, Math.min(height - r, event.y));
    }}
    function dragended(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }}
  </script>
</body>
</html>"""
