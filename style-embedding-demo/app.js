// Font style similarity demo.
const BIN_URL = "font_index.bin";
const JSON_URL = "font_index.json";
const TOP_K = 5;
const DEMO_PHRASE = "The quick brown fox";

const input = document.getElementById("font-search");
const listbox = document.getElementById("font-listbox");
const resultsSection = document.getElementById("results");
const resultsList = document.getElementById("results-list");
const resultsQuery = document.getElementById("results-query");
const status = document.getElementById("status");
const selectedSection = document.getElementById("selected");
const selectedQuery = document.getElementById("selected-query");
const selectedSample = document.getElementById("selected-sample");

// Data state.
let dim = 0;
let count = 0;
let labels = [];
let data = null; // Float32Array of length count * dim, row-major.

// Combobox state: `entries` is the full font list (one per font file, keyed by
// database index), `filtered` is the current (possibly filtered) subset shown
// in the listbox.
let entries = []; // [{ index, family, filename, italic, display }]
let filtered = [];
let activeOption = -1; // index into `filtered`.

function basename(path) {
  return path.split("/").pop();
}

function isItalic(filename) {
  return filename.includes("Italic");
}

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

const loadedFonts = new Set();

// Inject a Google Fonts stylesheet for *family* (both roman and italic) if not
// already loaded.
function loadFont(family) {
  if (loadedFonts.has(family)) return;
  loadedFonts.add(family);

  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(family)}:ital@0;1&display=swap`;
  document.head.appendChild(link);
}

function setSampleFont(el, family, italic) {
  el.style.fontFamily = `"${family}"`;
  el.style.fontStyle = italic ? "italic" : "normal";
}

async function load() {
  const [binResp, jsonResp] = await Promise.all([
    fetch(BIN_URL),
    fetch(JSON_URL),
  ]);

  if (!binResp.ok) {
    throw new Error(`Could not load ${BIN_URL} (HTTP ${binResp.status})`);
  }
  if (!jsonResp.ok) {
    throw new Error(`Could not load ${JSON_URL} (HTTP ${jsonResp.status})`);
  }

  const buffer = await binResp.arrayBuffer();
  const meta = await jsonResp.json();

  dim = meta.dim;
  count = meta.count;
  labels = meta.labels;
  data = new Float32Array(buffer);

  if (data.length !== count * dim) {
    throw new Error(`Expected ${count * dim} floats, got ${data.length}`);
  }

  buildEntries();
}

// One entry per font file (no family deduplication), keyed by database index so
// selecting an entry queries that exact vector. `display` is the combobox label.
function buildEntries() {
  entries = labels.map((label, i) => {
    const filename = basename(label.path);
    const italic = isItalic(filename);
    const display = italic ? `${label.family} (Italic)` : label.family;
    return { index: i, family: label.family, filename, italic, display };
  });

  entries.sort((a, b) => a.display.localeCompare(b.display));

  input.disabled = false;
  input.placeholder = `Search ${entries.length} fonts…`;
  setStatus("");
}

// Dot product over every stored vector. Vectors are centered + L2-normalized by
// the exporter, so this is exactly cosine similarity and needs no re-normalizing.
function search(queryIndex, topK) {
  const query = data.subarray(queryIndex * dim, (queryIndex + 1) * dim);
  const scores = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    let dot = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d++) {
      dot += data[base + d] * query[d];
    }
    scores[i] = dot;
  }

  return Array.from(scores)
    .map((score, i) => ({ score, index: i }))
    .filter((r) => r.index !== queryIndex) // self-exclusion
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((r) => ({ ...r, filename: basename(labels[r.index].path) }));
}

function renderResults(entry) {
  const top = search(entry.index, TOP_K);
  resultsQuery.textContent = entry.display;
  resultsList.innerHTML = "";

  top.forEach(({ score, index, filename }, rank) => {
    const family = labels[index].family;
    const italic = isItalic(filename);
    const li = document.createElement("li");

    const row = document.createElement("div");
    row.className = "row";

    const rankEl = document.createElement("span");
    rankEl.className = "rank";
    rankEl.textContent = String(rank + 1);

    const familyEl = document.createElement("span");
    familyEl.className = "family";
    familyEl.textContent = family;

    const fileEl = document.createElement("span");
    fileEl.className = "filename";
    fileEl.textContent = filename;

    const scoreEl = document.createElement("span");
    scoreEl.className = "score";
    scoreEl.textContent = score.toFixed(3);

    row.append(rankEl, familyEl, fileEl, scoreEl);

    const sample = document.createElement("div");
    sample.className = "sample";
    sample.textContent = DEMO_PHRASE;
    setSampleFont(sample, family, italic);
    loadFont(family);

    li.append(row, sample);
    resultsList.appendChild(li);
  });

  resultsSection.hidden = false;
}

function renderSelected(entry) {
  selectedQuery.textContent = entry.filename;
  selectedSample.textContent = DEMO_PHRASE;
  setSampleFont(selectedSample, entry.family, entry.italic);
  loadFont(entry.family);
  selectedSection.hidden = false;
}

// ── Combobox ────────────────────────────────────────────────────────────────

function openListbox() {
  listbox.hidden = false;
  input.setAttribute("aria-expanded", "true");
}

function closeListbox() {
  listbox.hidden = true;
  input.setAttribute("aria-expanded", "false");
  input.removeAttribute("aria-activedescendant");
}

function renderListbox() {
  listbox.innerHTML = "";

  if (filtered.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = input.value.trim() === "" ? "Type to search…" : "No matches";
    listbox.appendChild(li);
    return;
  }

  filtered.forEach((entry, i) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.id = `option-${i}`;
    li.textContent = entry.display;

    if (i === activeOption) {
      li.classList.add("active");
      input.setAttribute("aria-activedescendant", li.id);
    }

    // `mousedown` + `preventDefault` keeps focus on the input so the `blur`
    // handler doesn't close the list before the selection is applied.
    li.addEventListener("mousedown", (event) => {
      event.preventDefault();
      selectItem(i);
    });

    listbox.appendChild(li);
  });
}

function selectItem(i) {
  const entry = filtered[i];
  if (!entry) return;

  input.value = entry.display;
  activeOption = i;
  closeListbox();
  renderSelected(entry);
  renderResults(entry);
}

function refreshFilter() {
  const q = input.value.trim().toLowerCase();

  if (q === "") {
    filtered = [];
    activeOption = -1;
  } else {
    filtered = entries.filter((e) => e.display.toLowerCase().includes(q));
    activeOption = filtered.length ? 0 : -1;
  }

  renderListbox();
  openListbox();
}

function scrollActiveIntoView() {
  const el = listbox.querySelector("li.active");
  if (el) el.scrollIntoView({ block: "nearest" });
}

input.addEventListener("input", refreshFilter);
input.addEventListener("focus", refreshFilter);
input.addEventListener("blur", closeListbox);

input.addEventListener("keydown", (event) => {
  if (event.key === "ArrowDown") {
    event.preventDefault();
    if (!filtered.length) return;
    openListbox();
    activeOption = (activeOption + 1) % filtered.length;
    renderListbox();
    scrollActiveIntoView();
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    if (!filtered.length) return;
    activeOption = (activeOption - 1 + filtered.length) % filtered.length;
    renderListbox();
    scrollActiveIntoView();
  } else if (event.key === "Enter") {
    if (!listbox.hidden && activeOption >= 0) {
      event.preventDefault();
      selectItem(activeOption);
    }
  } else if (event.key === "Escape") {
    closeListbox();
  }
});

// ── Boot ────────────────────────────────────────────────────────────────────

load().catch((error) => {
  setStatus(error.message, true);
  console.error(error);
});
