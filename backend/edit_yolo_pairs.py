"""Local web editor for YOLO segmentation image/label pairs."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
import webbrowser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
DEFAULT_NAMES = {
    0: "table",
    1: "cue_ball",
    2: "eight_ball",
    3: "object_ball",
}
CLASS_COLORS = ("#00b87a", "#e6b800", "#8b5cf6", "#ef476f", "#2389da", "#f97316")


@dataclass(frozen=True)
class SegmentationInstance:
    class_id: int
    points: tuple[tuple[float, float], ...]
    line_number: int


@dataclass(frozen=True)
class ImageLabelPair:
    stem: str
    image: Path
    label: Path
    instances: tuple[SegmentationInstance, ...]


@dataclass(frozen=True)
class ViewerReport:
    pairs: int
    instances: int
    missing_labels: tuple[str, ...]
    missing_images: tuple[str, ...]
    errors: tuple[str, ...]
    output: Path


def parse_names_yaml(path: Path | None) -> dict[int, str]:
    """Read the simple numeric ``names`` mapping used by Ultralytics YAML files."""
    if path is None:
        return dict(DEFAULT_NAMES)
    names: dict[int, str] = {}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            match = re.match(r"^\s*(\d+)\s*:\s*['\"]?([^'\"#]+?)['\"]?\s*$", raw_line)
            if match:
                names[int(match.group(1))] = match.group(2).strip()
            elif not raw_line.startswith((" ", "\t")):
                break
    if not names:
        raise ValueError(f"No numeric class names were found in {path}")
    return names


def parse_segmentation_label(
    path: Path,
    class_names: dict[int, str],
) -> tuple[tuple[SegmentationInstance, ...], tuple[str, ...]]:
    """Parse and validate one Ultralytics YOLO segmentation label file."""
    instances: list[SegmentationInstance] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
        except (ValueError, IndexError):
            errors.append(f"{path}:{line_number}: class ID must be an integer")
            continue
        if class_id not in class_names:
            errors.append(f"{path}:{line_number}: unknown class ID {class_id}")
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: polygon coordinates must be numbers")
            continue
        if len(coordinates) < 6 or len(coordinates) % 2:
            errors.append(
                f"{path}:{line_number}: expected at least 3 x/y polygon pairs; "
                f"found {len(coordinates)} values"
            )
            continue
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            errors.append(f"{path}:{line_number}: coordinates must be normalized from 0 to 1")
            continue
        points = tuple(zip(coordinates[0::2], coordinates[1::2]))
        instances.append(SegmentationInstance(class_id, points, line_number))
    return tuple(instances), tuple(errors)


def _files_by_stem(directory: Path, suffixes: set[str]) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    errors: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            errors.append(
                f"Duplicate stem '{path.stem}' in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
            continue
        files[path.stem] = path
    return files, errors


def collect_pairs(
    images_dir: Path,
    labels_dir: Path,
    class_names: dict[int, str],
) -> tuple[list[ImageLabelPair], list[str], list[str], list[str]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    images, image_errors = _files_by_stem(images_dir, IMAGE_SUFFIXES)
    labels, label_errors = _files_by_stem(labels_dir, {".txt"})
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    errors = image_errors + label_errors
    pairs: list[ImageLabelPair] = []

    for stem in sorted(set(images).intersection(labels)):
        instances, parse_errors = parse_segmentation_label(labels[stem], class_names)
        errors.extend(parse_errors)
        pairs.append(ImageLabelPair(stem, images[stem], labels[stem], instances))
    return pairs, missing_labels, missing_images, errors


def _relative_url(path: Path, output: Path) -> str:
    relative = os.path.relpath(path.resolve(), output.parent.resolve()).replace(os.sep, "/")
    return quote(relative, safe="/._-")


def _polygon_markup(
    instance: SegmentationInstance,
    class_names: dict[int, str],
) -> str:
    class_name = class_names.get(instance.class_id, f"class_{instance.class_id}")
    color = CLASS_COLORS[instance.class_id % len(CLASS_COLORS)]
    points = " ".join(f"{x:.6f},{y:.6f}" for x, y in instance.points)
    title = html.escape(f"{instance.class_id} {class_name} · label line {instance.line_number}")
    return (
        f'<polygon points="{points}" style="--instance-color:{color}">'
        f"<title>{title}</title></polygon>"
    )


def _pair_card(pair: ImageLabelPair, output: Path, class_names: dict[int, str]) -> str:
    counts = Counter(instance.class_id for instance in pair.instances)
    badges = "".join(
        f'<span class="badge"><i style="--badge-color:{CLASS_COLORS[class_id % len(CLASS_COLORS)]}"></i>'
        f'{html.escape(class_names.get(class_id, f"class_{class_id}"))} × {count}</span>'
        for class_id, count in sorted(counts.items())
    ) or '<span class="badge">No instances</span>'
    polygons = "".join(_polygon_markup(item, class_names) for item in pair.instances)
    image_url = _relative_url(pair.image, output)
    return f"""
      <article class="pair-card">
        <header><strong>{html.escape(pair.image.name)}</strong><span>{len(pair.instances)} instances</span></header>
        <div class="frame">
          <img src="{image_url}" alt="{html.escape(pair.image.name)}" loading="lazy">
          <svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Segmentation polygons">
            {polygons}
          </svg>
        </div>
        <div class="badges">{badges}</div>
        <code>{html.escape(pair.label.name)}</code>
      </article>
    """


def _diagnostics_markup(
    missing_labels: Sequence[str],
    missing_images: Sequence[str],
    errors: Sequence[str],
) -> str:
    messages = [f"Missing label: {stem}.txt" for stem in missing_labels]
    messages.extend(f"Missing image for label: {stem}.txt" for stem in missing_images)
    messages.extend(errors)
    if not messages:
        return '<p class="valid">All discovered pairs and polygons are structurally valid.</p>'
    items = "".join(f"<li>{html.escape(message)}</li>" for message in messages)
    return f'<details open><summary>{len(messages)} dataset issue(s)</summary><ul>{items}</ul></details>'


def render_viewer(
    images_dir: Path,
    labels_dir: Path,
    output: Path,
    *,
    class_names: dict[int, str] | None = None,
) -> ViewerReport:
    """Create the standalone viewer and return its validation summary."""
    names = class_names or dict(DEFAULT_NAMES)
    pairs, missing_labels, missing_images, errors = collect_pairs(images_dir, labels_dir, names)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    instance_count = sum(len(pair.instances) for pair in pairs)
    legend = "".join(
        f'<span><i style="--badge-color:{CLASS_COLORS[class_id % len(CLASS_COLORS)]}"></i>'
        f'{class_id} {html.escape(name)}</span>'
        for class_id, name in sorted(names.items())
    )
    cards = "".join(_pair_card(pair, output, names) for pair in pairs)
    if not cards:
        cards = '<p class="empty">No matching image/label pairs were found.</p>'
    diagnostics = _diagnostics_markup(missing_labels, missing_images, errors)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YOLO segmentation pair viewer</title>
  <style>
    :root {{ color-scheme: light dark; --fill-opacity:.24; --page:#f5f7fa; --panel:#fff;
      --text:#15202b; --muted:#5b6875; --border:#d8dee6; --good:#087f5b; }}
    @media (prefers-color-scheme:dark) {{ :root {{ --page:#11161c; --panel:#1b222b;
      --text:#edf2f7; --muted:#aab6c2; --border:#384553; --good:#5ee0b1; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--page); color:var(--text);
      font:15px/1.45 system-ui,-apple-system,sans-serif; }}
    main {{ max-width:1500px; margin:auto; padding:24px; }}
    h1 {{ margin:0 0 4px; font-size:24px; }} .subtitle {{ color:var(--muted); margin:0 0 18px; }}
    .toolbar {{ display:flex; flex-wrap:wrap; align-items:center; gap:16px; padding:12px 14px;
      background:var(--panel); border:1px solid var(--border); border-radius:10px; }}
    .toolbar label {{ display:flex; gap:8px; align-items:center; }} input[type=range] {{ width:150px; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:12px; margin:14px 0; color:var(--muted); }}
    .legend span,.badge {{ display:inline-flex; align-items:center; gap:6px; }}
    i {{ width:11px; height:11px; border-radius:3px; background:var(--badge-color); display:inline-block; }}
    .valid {{ color:var(--good); }} details {{ border-left:4px solid #dc3545; padding-left:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr)); gap:18px; }}
    .pair-card {{ min-width:0; background:var(--panel); border:1px solid var(--border);
      border-radius:12px; overflow:hidden; }}
    .pair-card header {{ padding:10px 12px; display:flex; justify-content:space-between; gap:12px; }}
    .pair-card header span,.pair-card code {{ color:var(--muted); }}
    .frame {{ position:relative; background:#080b0e; }} .frame img {{ width:100%; height:auto; display:block; }}
    .frame svg {{ position:absolute; inset:0; width:100%; height:100%; overflow:visible; }}
    polygon {{ fill:var(--instance-color); fill-opacity:var(--fill-opacity); stroke:var(--instance-color);
      stroke-width:3; vector-effect:non-scaling-stroke; pointer-events:all; }}
    body.hide-fill polygon {{ fill-opacity:0; }} body.hide-overlay svg {{ display:none; }}
    .badges {{ display:flex; flex-wrap:wrap; gap:10px; padding:10px 12px 4px; }}
    .pair-card code {{ display:block; padding:5px 12px 12px; overflow-wrap:anywhere; font-size:12px; }}
    .empty {{ padding:24px; background:var(--panel); border:1px solid var(--border); border-radius:10px; }}
    @media (max-width:600px) {{ main {{ padding:14px; }} .toolbar {{ align-items:flex-start; flex-direction:column; }} }}
  </style>
</head>
<body>
  <main>
    <h1>YOLO segmentation pairs</h1>
    <p class="subtitle">{len(pairs)} paired images · {instance_count} instances</p>
    <section class="toolbar" aria-label="Overlay controls">
      <label>Mask opacity <input id="opacity" type="range" min="0" max="0.7" value="0.24" step="0.02"><output id="opacity-value">24%</output></label>
      <label><input id="fill" type="checkbox" checked> Show mask fill</label>
      <label><input id="overlay" type="checkbox" checked> Show polygons</label>
    </section>
    <div class="legend">{legend}</div>
    <section>{diagnostics}</section>
    <section class="grid">{cards}</section>
  </main>
  <script>
    const opacity = document.getElementById('opacity');
    const output = document.getElementById('opacity-value');
    opacity.addEventListener('input', () => {{
      document.documentElement.style.setProperty('--fill-opacity', opacity.value);
      output.value = `${{Math.round(Number(opacity.value) * 100)}}%`;
    }});
    document.getElementById('fill').addEventListener('change', event =>
      document.body.classList.toggle('hide-fill', !event.target.checked));
    document.getElementById('overlay').addEventListener('change', event =>
      document.body.classList.toggle('hide-overlay', !event.target.checked));
  </script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return ViewerReport(
        pairs=len(pairs),
        instances=instance_count,
        missing_labels=tuple(missing_labels),
        missing_images=tuple(missing_images),
        errors=tuple(errors),
        output=output,
    )


class EditorInstance(BaseModel):
    class_id: int
    points: list[tuple[float, float]]


class SaveLabelRequest(BaseModel):
    label_path: str
    instances: list[EditorInstance]


def _resolve_local_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _names_for_dataset(dataset: Path, data_path: str | None) -> tuple[dict[int, str], Path | None]:
    if data_path:
        selected = _resolve_local_path(data_path)
        return parse_names_yaml(selected), selected
    local_yaml = dataset / "dataset.yaml"
    if local_yaml.is_file():
        return parse_names_yaml(local_yaml), local_yaml
    return dict(DEFAULT_NAMES), None


def _editor_dataset_payload(dataset_value: str, split: str, data_path: str | None) -> dict:
    dataset = _resolve_local_path(dataset_value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
        raise HTTPException(status_code=400, detail="Split contains unsupported characters.")
    images_dir = dataset / "images" / split
    labels_dir = dataset / "labels" / split
    try:
        names, selected_yaml = _names_for_dataset(dataset, data_path)
        pairs, missing_labels, missing_images, errors = collect_pairs(images_dir, labels_dir, names)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "dataset": str(dataset),
        "split": split,
        "data_path": str(selected_yaml) if selected_yaml else "",
        "class_names": {str(class_id): name for class_id, name in sorted(names.items())},
        "pairs": [
            {
                "stem": pair.stem,
                "image_name": pair.image.name,
                "image_url": f"/api/image?path={quote(str(pair.image.resolve()), safe='')}",
                "label_path": str(pair.label.resolve()),
                "instances": [
                    {
                        "class_id": instance.class_id,
                        "points": instance.points,
                        "line_number": instance.line_number,
                    }
                    for instance in pair.instances
                ],
            }
            for pair in pairs
        ],
        "issues": {
            "missing_labels": missing_labels,
            "missing_images": missing_images,
            "errors": errors,
        },
    }


def _validate_editor_instances(instances: Sequence[EditorInstance]) -> None:
    for index, instance in enumerate(instances, 1):
        if instance.class_id < 0:
            raise HTTPException(status_code=400, detail=f"Instance {index} has a negative class ID.")
        if len(instance.points) < 3:
            raise HTTPException(status_code=400, detail=f"Instance {index} needs at least 3 points.")
        for x, y in instance.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise HTTPException(
                    status_code=400,
                    detail=f"Instance {index} contains a coordinate outside 0–1.",
                )


def _write_label_atomically(path: Path, instances: Sequence[EditorInstance]) -> None:
    lines = []
    for instance in instances:
        coordinates = " ".join(f"{value:.6f}" for point in instance.points for value in point)
        lines.append(f"{instance.class_id} {coordinates}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(temporary, path)


def _editor_document(default_dataset: Path, default_split: str, default_data: Path | None) -> str:
    dataset_json = json.dumps(str(default_dataset.resolve())).replace("</", "<\\/")
    split_json = json.dumps(default_split).replace("</", "<\\/")
    data_json = json.dumps(str(default_data.resolve()) if default_data else "").replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YOLO segmentation editor</title>
  <style>
    :root {{ color-scheme:light dark; --bg:#f3f5f8; --panel:#fff; --text:#17202a;
      --muted:#647180; --border:#d5dce5; --accent:#1677ff; --danger:#d9363e; }}
    @media(prefers-color-scheme:dark) {{ :root {{ --bg:#10151b; --panel:#1a222c; --text:#ecf2f8;
      --muted:#a6b2bf; --border:#364351; --accent:#63a7ff; --danger:#ff7077; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text);
      font:14px/1.4 system-ui,-apple-system,sans-serif; }}
    button,input,select {{ font:inherit; }} button {{ cursor:pointer; }}
    .top {{ padding:12px; background:var(--panel); border-bottom:1px solid var(--border); }}
    .row {{ display:flex; gap:8px; flex-wrap:wrap; align-items:end; }}
    label {{ display:grid; gap:4px; color:var(--muted); }} input,select {{ color:var(--text);
      background:var(--bg); border:1px solid var(--border); border-radius:6px; padding:7px 8px; }}
    #dataset-path {{ min-width:min(520px,80vw); }} #data-path {{ min-width:min(360px,70vw); }}
    button {{ color:var(--text); background:var(--panel); border:1px solid var(--border);
      border-radius:6px; padding:7px 10px; }} button.primary {{ background:var(--accent); color:white;
      border-color:var(--accent); }} button.danger {{ color:var(--danger); }} button:disabled {{ opacity:.45; cursor:not-allowed; }}
    details {{ margin-top:10px; }} details .row {{ padding-top:8px; }}
    #status {{ min-height:22px; padding:7px 12px; color:var(--muted); }} #status.error {{ color:var(--danger); }}
    .layout {{ display:grid; grid-template-columns:260px minmax(0,1fr); height:calc(100vh - 166px); min-height:520px; }}
    .sidebar {{ border-right:1px solid var(--border); background:var(--panel); overflow:auto; padding:8px; }}
    .pair-button {{ display:block; width:100%; text-align:left; margin-bottom:5px; overflow:hidden;
      text-overflow:ellipsis; white-space:nowrap; }} .pair-button.active {{ border-color:var(--accent); color:var(--accent); }}
    .workspace {{ min-width:0; display:flex; flex-direction:column; }} .tools {{ display:flex; gap:7px;
      align-items:center; flex-wrap:wrap; padding:8px; background:var(--panel); border-bottom:1px solid var(--border); }}
    .tools .spacer {{ flex:1; }} .canvas-wrap {{ min-height:0; flex:1; overflow:auto; display:grid;
      place-items:center; padding:14px; background:#090c10; }}
    .frame {{ position:relative; display:inline-block; max-width:100%; line-height:0; user-select:none; }}
    .frame img {{ display:block; max-width:100%; max-height:calc(100vh - 270px); width:auto; height:auto; }}
    .frame svg {{ position:absolute; inset:0; width:100%; height:100%; cursor:default; }}
    polygon {{ fill:var(--color); fill-opacity:.23; stroke:var(--color); stroke-width:3;
      vector-effect:non-scaling-stroke; cursor:pointer; }} polygon.selected {{ fill-opacity:.38; stroke-width:5; }}
    polyline.draft {{ fill:none; stroke:#fff; stroke-width:3; vector-effect:non-scaling-stroke; }}
    circle.handle {{ fill:#fff; stroke:#111; stroke-width:2; vector-effect:non-scaling-stroke; cursor:grab; }}
    .footer {{ padding:7px 10px; color:var(--muted); background:var(--panel); border-top:1px solid var(--border); }}
    .empty {{ color:#dce5ef; text-align:center; padding:30px; }} kbd {{ border:1px solid var(--border); border-radius:4px; padding:1px 4px; }}
    @media(max-width:760px) {{ .layout {{ grid-template-columns:1fr; height:auto; }} .sidebar {{ max-height:150px;
      border-right:0; border-bottom:1px solid var(--border); }} .canvas-wrap {{ min-height:420px; }} }}
  </style>
</head>
<body>
  <section class="top">
    <div class="row">
      <label>Dataset root<input id="dataset-path"></label>
      <label>Split<input id="split" value="train" size="8"></label>
      <label>Class YAML (optional)<input id="data-path"></label>
      <button id="load" class="primary">Load dataset</button>
    </div>
    <details>
      <summary>Create a new dataset from selected images</summary>
      <div class="row">
        <label>New dataset path<input id="new-path" placeholder="datasets/my_dataset"></label>
        <label>Split<input id="new-split" value="train" size="8"></label>
        <label>Images<input id="new-images" type="file" accept="image/*,.heic,.heif" multiple></label>
        <button id="create">Create dataset</button>
      </div>
    </details>
  </section>
  <div id="status"></div>
  <main class="layout">
    <aside class="sidebar"><div id="pairs"></div></aside>
    <section class="workspace">
      <div class="tools">
        <button id="prev">← Previous</button><button id="next">Next →</button>
        <label>Class<select id="class-select"></select></label>
        <button id="select-mode">Select/edit</button><button id="add-mode">Add polygon</button>
        <button id="finish" disabled>Finish polygon</button><button id="cancel" disabled>Cancel</button>
        <button id="delete" class="danger" disabled>Delete selected</button>
        <span class="spacer"></span><button id="save" class="primary" disabled>Save label</button>
      </div>
      <div class="canvas-wrap"><div id="empty" class="empty">Load a dataset to begin.</div>
        <div id="frame" class="frame" hidden><img id="image"><svg id="overlay" viewBox="0 0 1 1" preserveAspectRatio="none"></svg></div>
      </div>
      <div id="footer" class="footer">Click a polygon to select it. Drag white handles to edit vertices.</div>
    </section>
  </main>
  <script>
    const DEFAULT_DATASET={dataset_json}, DEFAULT_SPLIT={split_json}, DEFAULT_DATA={data_json};
    const COLORS=['#00b87a','#e6b800','#8b5cf6','#ef476f','#2389da','#f97316'];
    const $=id=>document.getElementById(id);
    let dataset=null, index=0, instances=[], selected=-1, draft=[], mode='select', dirty=false, drag=null;
    $('dataset-path').value=DEFAULT_DATASET; $('split').value=DEFAULT_SPLIT; $('data-path').value=DEFAULT_DATA;
    function status(message,error=false) {{ $('status').textContent=message; $('status').classList.toggle('error',error); }}
    function escapeHtml(value) {{ const node=document.createElement('span'); node.textContent=String(value); return node.innerHTML; }}
    async function api(url,options) {{ const response=await fetch(url,options); const data=await response.json().catch(()=>({{}}));
      if(!response.ok) throw new Error(data.detail||`Request failed (${{response.status}})`); return data; }}
    function query() {{ return new URLSearchParams({{path:$('dataset-path').value,split:$('split').value,data:$('data-path').value}}); }}
    async function loadDataset() {{ if(dirty&&!confirm('Discard unsaved label changes?')) return; try {{
      status('Loading…'); dataset=await api('/api/dataset?'+query()); index=0; dirty=false; renderClasses(); renderPairList(); openPair(0);
      const issues=dataset.issues.missing_labels.length+dataset.issues.missing_images.length+dataset.issues.errors.length;
      status(`${{dataset.pairs.length}} pairs loaded · ${{issues}} structural issues` ,issues>0);
    }} catch(error) {{ status(error.message,true); }} }}
    function renderClasses() {{ $('class-select').innerHTML=Object.entries(dataset.class_names).map(([id,name])=>
      `<option value="${{escapeHtml(id)}}">${{escapeHtml(id)}} ${{escapeHtml(name)}}</option>`).join(''); }}
    function renderPairList() {{ $('pairs').innerHTML=dataset.pairs.map((pair,i)=>
      `<button class="pair-button ${{i===index?'active':''}}" data-index="${{i}}">${{escapeHtml(pair.image_name)}} · ${{pair.instances.length}}</button>`).join('')||'No paired files';
      document.querySelectorAll('.pair-button').forEach(button=>button.onclick=()=>switchPair(Number(button.dataset.index))); }}
    function switchPair(next) {{ if(dirty&&!confirm('Discard unsaved label changes?')) return; openPair(next); }}
    function openPair(next) {{ if(!dataset||!dataset.pairs.length) {{ $('frame').hidden=true; $('empty').hidden=false; return; }}
      index=Math.max(0,Math.min(next,dataset.pairs.length-1)); const pair=dataset.pairs[index];
      instances=structuredClone(pair.instances); selected=-1; draft=[]; mode='select'; dirty=false;
      $('image').src=pair.image_url+'&v='+Date.now(); $('frame').hidden=false; $('empty').hidden=true; renderPairList(); render(); }}
    function polygonPoints(points) {{ return points.map(point=>`${{point[0]}},${{point[1]}}`).join(' '); }}
    function svgElement(name,attributes={{}}) {{ const node=document.createElementNS('http://www.w3.org/2000/svg',name);
      for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,value); return node; }}
    function render() {{ const svg=$('overlay'); svg.replaceChildren(); instances.forEach((instance,i)=>{{
      const polygon=svgElement('polygon',{{points:polygonPoints(instance.points),style:`--color:${{COLORS[instance.class_id%COLORS.length]}}`}});
      if(i===selected) polygon.classList.add('selected'); polygon.onclick=event=>{{event.stopPropagation(); if(mode==='select'){{selected=i;render();}}}};
      const title=svgElement('title'); title.textContent=`${{instance.class_id}} ${{dataset.class_names[instance.class_id]||'unknown'}}`; polygon.append(title); svg.append(polygon);
      if(i===selected&&mode==='select') instance.points.forEach((point,vertex)=>{{ const handle=svgElement('circle',{{cx:point[0],cy:point[1],r:.008,class:'handle'}});
        handle.onpointerdown=event=>{{event.preventDefault();event.stopPropagation();drag={{instance:i,vertex}};handle.setPointerCapture(event.pointerId);}}; svg.append(handle); }});
    }}); if(draft.length) {{ svg.append(svgElement('polyline',{{points:polygonPoints(draft),class:'draft'}})); draft.forEach(point=>svg.append(svgElement('circle',{{cx:point[0],cy:point[1],r:.006,class:'handle'}}))); }}
      $('finish').disabled=draft.length<3; $('cancel').disabled=!draft.length; $('delete').disabled=selected<0; $('save').disabled=!dirty;
      const pair=dataset&&dataset.pairs[index]; $('footer').textContent=pair?`${{pair.image_name}} · ${{instances.length}} instances${{dirty?' · unsaved changes':''}}`:'No pair selected'; }}
    function position(event) {{ const box=$('overlay').getBoundingClientRect(); return [Math.max(0,Math.min(1,(event.clientX-box.left)/box.width)),Math.max(0,Math.min(1,(event.clientY-box.top)/box.height))]; }}
    $('overlay').onclick=event=>{{ if(mode==='add'){{ draft.push(position(event));dirty=true;render(); }} else {{selected=-1;render();}} }};
    window.addEventListener('pointermove',event=>{{ if(!drag)return; instances[drag.instance].points[drag.vertex]=position(event);dirty=true;render(); }});
    window.addEventListener('pointerup',()=>drag=null);
    function finishDraft() {{ if(draft.length<3)return; instances.push({{class_id:Number($('class-select').value),points:draft}}); selected=instances.length-1;draft=[];mode='select';dirty=true;render(); }}
    $('finish').onclick=finishDraft; $('cancel').onclick=()=>{{draft=[];mode='select';render();}};
    $('add-mode').onclick=()=>{{mode='add';selected=-1;draft=[];render();status('Add mode: click at least 3 boundary points, then Finish polygon.');}};
    $('select-mode').onclick=()=>{{mode='select';draft=[];render();}};
    $('delete').onclick=()=>{{if(selected>=0){{instances.splice(selected,1);selected=-1;dirty=true;render();}}}};
    $('prev').onclick=()=>switchPair(index-1); $('next').onclick=()=>switchPair(index+1); $('load').onclick=loadDataset;
    $('save').onclick=async()=>{{try{{const pair=dataset.pairs[index];await api('/api/labels/save',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{label_path:pair.label_path,instances}})}});pair.instances=structuredClone(instances);dirty=false;render();renderPairList();status(`Saved ${{pair.label_path}}`);}}catch(error){{status(error.message,true);}}}};
    $('create').onclick=async()=>{{try{{const files=$('new-images').files;if(!files.length)throw new Error('Select at least one image.');const form=new FormData();form.append('dataset_path',$('new-path').value);form.append('split',$('new-split').value);for(const file of files)form.append('files',file);status('Creating dataset…');const result=await api('/api/datasets/create',{{method:'POST',body:form}});$('dataset-path').value=result.dataset;$('split').value=result.split;$('data-path').value=result.data_path;await loadDataset();}}catch(error){{status(error.message,true);}}}};
    window.addEventListener('keydown',event=>{{if(event.key==='Enter'&&mode==='add')finishDraft();if((event.key==='Delete'||event.key==='Backspace')&&selected>=0&&!['INPUT','SELECT'].includes(document.activeElement.tagName)){{$('delete').click();event.preventDefault();}}}});
    window.addEventListener('beforeunload',event=>{{if(dirty){{event.preventDefault();event.returnValue='';}}}}); loadDataset();
  </script>
</body>
</html>"""


def create_editor_app(
    default_dataset: Path,
    default_split: str = "train",
    default_data: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="YOLO Segmentation Editor", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def editor_page() -> str:
        return _editor_document(default_dataset, default_split, default_data)

    @app.get("/api/dataset")
    def dataset_payload(path: str, split: str = "train", data: str = "") -> dict:
        return _editor_dataset_payload(path, split, data or None)

    @app.get("/api/image")
    def dataset_image(path: str) -> FileResponse:
        image = _resolve_local_path(path)
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=404, detail="Image not found or unsupported.")
        return FileResponse(image, headers={"Cache-Control": "no-store"})

    @app.post("/api/labels/save")
    def save_label(request: SaveLabelRequest) -> dict:
        label = _resolve_local_path(request.label_path)
        if label.suffix.lower() != ".txt" or "labels" not in label.parts or not label.parent.is_dir():
            raise HTTPException(status_code=400, detail="Refusing to write outside an existing labels directory.")
        _validate_editor_instances(request.instances)
        _write_label_atomically(label, request.instances)
        return {"saved": str(label), "instances": len(request.instances)}

    @app.post("/api/datasets/create")
    async def create_dataset(
        dataset_path: str = Form(...),
        split: str = Form("train"),
        files: list[UploadFile] = File(...),
    ) -> dict:
        dataset = _resolve_local_path(dataset_path)
        if dataset in {Path("/").resolve(), Path.home().resolve()}:
            raise HTTPException(status_code=400, detail="Choose a dedicated dataset directory.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
            raise HTTPException(status_code=400, detail="Split contains unsupported characters.")
        uploads: list[tuple[str, bytes]] = []
        for upload in files:
            name = Path(upload.filename or "").name
            if not name or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"Unsupported image: {name or 'unnamed'}")
            content = await upload.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Empty image: {name}")
            if len(content) > 100 * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"Image exceeds 100 MB: {name}")
            uploads.append((name, content))
        if len({name for name, _ in uploads}) != len(uploads):
            raise HTTPException(status_code=400, detail="Selected images contain duplicate filenames.")

        for standard_split in ("train", "val", "test"):
            (dataset / "images" / standard_split).mkdir(parents=True, exist_ok=True)
            (dataset / "labels" / standard_split).mkdir(parents=True, exist_ok=True)
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        collisions = [name for name, _ in uploads if (images_dir / name).exists()]
        if collisions:
            raise HTTPException(status_code=409, detail=f"Images already exist: {', '.join(collisions)}")
        for name, content in uploads:
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            destination = images_dir / name
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
            label = labels_dir / f"{Path(name).stem}.txt"
            if not label.exists():
                label.write_text("", encoding="utf-8")

        data_file = dataset / "dataset.yaml"
        if not data_file.exists():
            data_file.write_text(
                f"path: {json.dumps(str(dataset))}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                "names:\n  0: table\n  1: cue_ball\n  2: eight_ball\n  3: object_ball\n",
                encoding="utf-8",
            )
        return {
            "dataset": str(dataset),
            "split": split,
            "data_path": str(data_file),
            "created": len(uploads),
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    backend_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="View and edit YOLO segmentation image/label pairs in a local web app."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=backend_dir / "datasets" / "rail_kick_example",
        help="Dataset root containing images/<split> and labels/<split>.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to view (default: train).")
    parser.add_argument("--data", type=Path, help="Optional dataset YAML providing the class names.")
    parser.add_argument("--host", default="127.0.0.1", help="Editor host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Editor port (default: 8765).")
    parser.add_argument("--open", action="store_true", help="Open the editor in the default browser.")
    parser.add_argument(
        "--export",
        type=Path,
        metavar="HTML",
        help="Export the split as a read-only standalone HTML file instead of starting the editor.",
    )
    parser.add_argument("--strict", action="store_true", help="With --export, exit nonzero for dataset issues.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.export:
        images_dir = args.dataset / "images" / args.split
        labels_dir = args.dataset / "labels" / args.split
        names, _ = _names_for_dataset(args.dataset.resolve(), str(args.data) if args.data else None)
        report = render_viewer(images_dir, labels_dir, args.export, class_names=names)
        print(f"Viewer: {report.output}")
        print(f"Pairs: {report.pairs}; instances: {report.instances}")
        issue_count = len(report.missing_labels) + len(report.missing_images) + len(report.errors)
        print(f"Issues: {issue_count}")
        if args.open:
            webbrowser.open(report.output.as_uri())
        return 1 if args.strict and issue_count else 0

    url = f"http://{args.host}:{args.port}"
    print(f"YOLO editor: {url}")
    if args.open:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    uvicorn.run(
        create_editor_app(args.dataset, args.split, args.data),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
