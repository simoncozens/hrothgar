/* Glyph review rig.
 *
 * Loads review_manifest.json (static structure) and review_state.json
 * (glyph key -> category).  Labels are held in `state` and can be downloaded
 * as review_state.json, which the user drops back into the served directory.
 */

const CATEGORIES = [
  "acceptable",
  "ambiguous",
  "bad-skeleton",
  "wrong-style",
  "bad-execution",
  "wrong-content",
];

const SPECIMEN_LINES = [
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "abcdefghijklmnopqrstuvwxyz",
  "0123456789",
];

let families = [];
let state = {}; // "<family>/U+XXXX" -> category
let currentFamilyIndex = -1;
let currentGlyphIndex = 0;
let fontLoadedOk = true;
let renderSeq = 0; // guards against overlapping async family renders

const els = {
  progressLabel: document.getElementById("progress-label"),
  progressFill: document.getElementById("progress-fill"),
  familyTitle: document.getElementById("family-title"),
  specimen: document.getElementById("specimen"),
  hint: document.getElementById("hint"),
  list: document.getElementById("glyph-list"),
  download: document.getElementById("download-btn"),
  prevFamily: document.getElementById("prev-family"),
  nextFamily: document.getElementById("next-family"),
};

// ── Helpers ───────────────────────────────────────────────────────────────

function hex(cp) {
  return cp.toString(16).toUpperCase().padStart(4, "0");
}

function glyphKey(family, cp) {
  return `${family.key}/U+${hex(cp)}`;
}

function isComplete(family) {
  return family.glyphs.every((g) => state[glyphKey(family, g.cp)]);
}

function completedCount() {
  return families.filter(isComplete).length;
}

async function loadJson(url, fallback) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return fallback;
    return await response.json();
  } catch (err) {
    console.warn(`Could not load ${url}`, err);
    return fallback;
  }
}

// ── Progress ──────────────────────────────────────────────────────────────

function updateProgress() {
  const done = completedCount();
  const total = families.length;
  const pct = total ? (100 * done) / total : 0;
  els.progressLabel.textContent =
    `Families: ${done} / ${total} (${pct.toFixed(1)}%)`;
  els.progressFill.style.width = `${pct}%`;
  document.title =
    `Glyph review rig — ${Object.keys(state).length} labels · ` +
    `${done} / ${total} families`;
}

// ── Font loading (Google Fonts CSS) ───────────────────────────────────────

function buildFontCssUrl(family) {
  // Spaces become '+'; everything else is URL-encoded.  Instances are at
  // Regular (400); the two single-weight outliers (Buda 300, UnifrakturCook
  // 700) use their own weight so the preview still resolves.
  const encoded = encodeURIComponent(family.family).replace(/%20/g, "+");
  const weight = family.weight || 400;
  const axes =
    family.style === "italic"
      ? `ital,wght@1,${weight}`
      : `wght@${weight}`;
  return `https://fonts.googleapis.com/css2?family=${encoded}:${axes}&display=swap`;
}

function loadFamilyFont(family) {
  return new Promise((resolve) => {
    const previous = document.getElementById("family-font-link");
    if (previous) previous.remove();

    const link = document.createElement("link");
    link.id = "family-font-link";
    link.rel = "stylesheet";
    link.href = buildFontCssUrl(family);
    link.onload = async () => {
      const italic = family.style === "italic" ? "italic " : "";
      const spec = `${italic}${family.weight || 400} 48px "${family.family}"`;
      try {
        await document.fonts.load(spec);
      } catch (err) {
        /* ignore: check() below tells us the truth */
      }
      resolve(document.fonts.check(spec));
    };
    link.onerror = () => resolve(false);
    document.head.appendChild(link);
  });
}

// ── Rendering ─────────────────────────────────────────────────────────────

function renderGlyphRow(family, index) {
  const g = family.glyphs[index];
  const key = glyphKey(family, g.cp);

  const row = document.createElement("div");
  row.className = "glyph-row";
  row.dataset.index = String(index);
  row.dataset.key = key;
  row.dataset.label = state[key] || "";

  // Ground truth + generated, side by side in the same 384x256 em frame.
  const pair = document.createElement("div");
  pair.className = "pair";

  const gtFigure = document.createElement("figure");
  const gtCaption = document.createElement("figcaption");
  gtCaption.textContent = "ground truth";
  const gtBox = document.createElement("div");
  gtBox.className = "box";
  gtBox.title = `${family.family} (${family.style}, ${family.weight})`;
  if (g.has_gt === true) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 384 256");
    const text = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "text"
    );
    text.setAttribute("x", "64"); // origin_x = 0.5em * 128
    text.setAttribute("y", "192"); // baseline = 1.5em * 128
    text.setAttribute("font-size", "128"); // 1em = 128 px
    text.setAttribute("class", "gt-text");
    text.textContent = String.fromCodePoint(g.cp);
    svg.appendChild(text);
    gtBox.appendChild(svg);
  } else {
    gtBox.classList.add("gt-missing");
    gtBox.textContent = g.has_gt === false ? "not in font" : "GT unknown";
  }
  gtFigure.appendChild(gtCaption);
  gtFigure.appendChild(gtBox);

  const genFigure = document.createElement("figure");
  const genCaption = document.createElement("figcaption");
  genCaption.textContent = "generated";
  const genBox = document.createElement("div");
  genBox.className = "box";
  const img = document.createElement("img");
  img.src = g.file;
  img.alt = `generated ${key}`;
  img.loading = "lazy";
  genBox.appendChild(img);
  genFigure.appendChild(genCaption);
  genFigure.appendChild(genBox);

  pair.appendChild(gtFigure);
  pair.appendChild(genFigure);

  // Codepoint, radios, triage note.
  const meta = document.createElement("div");
  meta.className = "meta";

  const cpLabel = document.createElement("div");
  cpLabel.className = "cp-label";
  const char = g.has_gt ? String.fromCodePoint(g.cp) : "∅";
  cpLabel.textContent = `U+${hex(g.cp)}  ${char}`;
  meta.appendChild(cpLabel);

  const radios = document.createElement("div");
  radios.className = "radios";
  CATEGORIES.forEach((category, ci) => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = `g-${key}`;
    input.value = category;
    input.checked = state[key] === category;
    input.addEventListener("change", () => assign(family, index, category));
    label.appendChild(input);
    label.appendChild(document.createTextNode(` ${ci + 1}. ${category}`));
    radios.appendChild(label);
  });
  meta.appendChild(radios);

  if (g.triage != null || g.iou != null || g.hausdorff != null) {
    const triage = document.createElement("div");
    triage.className = "triage";
    const bits = [];
    if (g.triage) bits.push(`triage: ${g.triage}`);
    if (g.iou != null) bits.push(`iou ${g.iou.toFixed(2)}`);
    if (g.hausdorff != null) bits.push(`haus ${g.hausdorff.toFixed(1)}`);
    triage.textContent = bits.join("  ·  ");
    meta.appendChild(triage);
  }

  row.appendChild(pair);
  row.appendChild(meta);
  return row;
}

async function renderFamily(index) {
  const family = families[index];
  if (!family) return;
  const seq = ++renderSeq;
  currentFamilyIndex = index;

  // Start at the first untagged glyph (or 0 if everything is tagged).
  const firstUntagged = family.glyphs.findIndex(
    (g) => !state[glyphKey(family, g.cp)]
  );
  currentGlyphIndex = firstUntagged >= 0 ? firstUntagged : 0;

  els.familyTitle.textContent =
    `${family.family}  ·  ${family.key}  (${family.style}, ${family.weight})`;

  document.documentElement.style.setProperty(
    "--glyph-font",
    `"${family.family}"`
  );
  document.documentElement.style.setProperty(
    "--glyph-weight",
    String(family.weight || 400)
  );
  document.documentElement.style.setProperty(
    "--glyph-style",
    family.style === "italic" ? "italic" : "normal"
  );

  els.specimen.textContent = "";
  for (const line of SPECIMEN_LINES) {
    const div = document.createElement("div");
    div.textContent = line;
    els.specimen.appendChild(div);
  }

  els.list.replaceChildren();
  for (let i = 0; i < family.glyphs.length; i++) {
    els.list.appendChild(renderGlyphRow(family, i));
  }
  updateProgress();
  els.hint.textContent = "Loading font preview…";

  fontLoadedOk = await loadFamilyFont(family);
  if (seq !== renderSeq) return; // a newer family render won; drop this one
  if (!fontLoadedOk) {
    els.hint.textContent =
      "⚠ Could not load this family from Google Fonts — the ground-truth " +
      "column is falling back to a system font.";
  } else {
    els.hint.textContent =
      "1–6 assign a category · ↑/↓ or j/k move · auto-advances to the next " +
      "untagged glyph";
  }
  scrollToCurrent();
}

// ── Labelling ─────────────────────────────────────────────────────────────

function scrollToCurrent() {
  els.list.querySelectorAll(".glyph-row.current").forEach((row) => {
    row.classList.remove("current");
  });
  const row = els.list.querySelector(
    `.glyph-row[data-index="${currentGlyphIndex}"]`
  );
  if (row) {
    row.classList.add("current");
    row.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

function assign(family, index, category) {
  const g = family.glyphs[index];
  const key = glyphKey(family, g.cp);
  state[key] = category;
  currentGlyphIndex = index;

  // Reflect the choice in the row's radios (assign may be called by keyboard).
  const row = els.list.querySelector(
    `.glyph-row[data-index="${index}"]`
  );
  if (row) {
    row.dataset.label = category;
    row.querySelectorAll('input[type="radio"]').forEach((input) => {
      input.checked = input.value === category;
    });
  }

  updateProgress();

  if (isComplete(family)) {
    advanceToNextFamily();
  } else {
    advanceToNextGlyph(family, index);
  }
}

function advanceToNextGlyph(family, fromIndex) {
  const n = family.glyphs.length;
  for (let step = 1; step <= n; step++) {
    const j = (fromIndex + step) % n;
    if (!state[glyphKey(family, family.glyphs[j].cp)]) {
      currentGlyphIndex = j;
      scrollToCurrent();
      return;
    }
  }
}

function advanceToNextFamily() {
  const total = families.length;
  for (let step = 1; step <= total; step++) {
    const j = (currentFamilyIndex + step) % total;
    if (!isComplete(families[j])) {
      renderFamily(j);
      return;
    }
  }
  els.hint.textContent = "🎉 All families are complete.";
}

function stepFamily(delta) {
  const total = families.length;
  if (!total) return;
  const j = (currentFamilyIndex + delta + total) % total;
  renderFamily(j);
}

// ── Events ────────────────────────────────────────────────────────────────

els.prevFamily.addEventListener("click", () => stepFamily(-1));
els.nextFamily.addEventListener("click", () => stepFamily(1));

els.download.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(state, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "review_state.json";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

document.addEventListener("keydown", (event) => {
  const family = families[currentFamilyIndex];
  if (!family) return;

  // Ignore modifier combos so browser shortcuts still work.
  if (event.ctrlKey || event.metaKey || event.altKey) return;

  if (event.key >= "1" && event.key <= "6") {
    const category = CATEGORIES[Number(event.key) - 1];
    assign(family, currentGlyphIndex, category);
    event.preventDefault();
    return;
  }

  if (event.key === "ArrowDown" || event.key === "j") {
    currentGlyphIndex = (currentGlyphIndex + 1) % family.glyphs.length;
    scrollToCurrent();
    event.preventDefault();
    return;
  }

  if (event.key === "ArrowUp" || event.key === "k") {
    currentGlyphIndex =
      (currentGlyphIndex - 1 + family.glyphs.length) % family.glyphs.length;
    scrollToCurrent();
    event.preventDefault();
  }
});

// ── Boot ──────────────────────────────────────────────────────────────────

async function init() {
  const manifest = await loadJson("review_manifest.json", null);
  if (!manifest) {
    els.progressLabel.textContent =
      "Could not load review_manifest.json — run build_review_rig.py first.";
    return;
  }
  families = manifest.families || [];
  state = await loadJson("review_state.json", {});

  updateProgress();

  let start = families.findIndex((family) => !isComplete(family));
  if (start < 0) start = 0;
  await renderFamily(start);
}

init();
