import { readFile } from 'node:fs/promises';

function get(path) {
    // We'll change this to fetch when we move to web
    return readFile(path)
}

const buf = (await get('outputs/style_embedder/font_index.bin')).buffer;
const json = JSON.parse(await get('outputs/style_embedder/font_index.json'));
const N = json["count"];
const D = json["dim"];
const data = new Float32Array(buf); // 3000*256 floats, row-major

function search(query, topK = 10) {
  // pre-normalize query if embeddings are pre-normalized (recommended)
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let dot = 0;
    const base = i * D;
    for (let d = 0; d < D; d++) dot += data[base + d] * query[d];
    scores[i] = dot; // = cosine similarity if both sides are unit-normalized
  }
  return Array.from(scores)
    .map((s, i) => [s, i])
    .sort((a, b) => b[0] - a[0])
      .slice(0, topK)
      .map(([s, i]) => [s, json.labels[i].family]);
}

function findFamily(familyName) {
    return json.labels.findIndex(item => item.family == familyName);
}

function nearest(familyName) {
    let index = findFamily(familyName);
    if (index == -1) { return [] }
    let embedding = data.slice(index * D, index * D + D);
    let results = search(embedding);
    // Skip self
    if (results[0][1] == familyName) { return results.slice(1) }
    return results;
}

console.log(nearest("Gulzar"));
