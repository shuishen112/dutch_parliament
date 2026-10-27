// Search over the slavery-related Dutch parliamentary papers, 1814–1873.
// Adapted from historical_nlp/www_experiments/site/app.js (the dwi-newspapers site):
// same index files, tokenization, fold(), adaptive OCR threshold and scoring. Added here:
//   * {a|b|c} word groups, * wildcards and ~N unordered windows in typed queries;
//   * passages are overlapping slices of long documents, so a match is only kept in
//     the passage that owns its first word (see export_site.py);
//   * precomputed concept searches (the free-womb rules of stage1_candidates.py).
"use strict";

const DATA = "data/";
const PAGE_SIZE = 20;
const LABELS = [
  ["direct", "Yes, directly", 2],
  ["related", "Related", 1],
  ["no", "No", 0],
  ["unsure", "Unsure", ""],
];

const state = {
  meta: null,
  vocab: new Map(),     // length -> {terms, arrays, df: Map}
  postShards: new Map(), // shard -> Map(term -> raw postings)
  textShards: new Map(), // shard -> Map(pid -> {page, text})
  concepts: new Map(),   // key -> rows of the precomputed concept file
  hits: [],
  shown: 0,
  expansions: [],
  excluded: new Set(),
  tiersOff: new Set(),
  lastQuery: null,
};

function storagePrefix() {
  return (state.meta && state.meta.storage_prefix) || "nl-parliament";
}
function reviewKey() { return storagePrefix() + "-reviews-v1"; }
function ocrKey() { return storagePrefix() + "-ocr-v1"; }

// ------------------------------------------------------------------ text
const TOKEN_RE = /[\p{L}\p{N}]+/gu;
const fold = (s) => s.toLowerCase().normalize("NFKD").replace(/\p{M}/gu, "");
function tokenize(text) {
  const out = [];
  for (const m of text.matchAll(TOKEN_RE)) out.push([m.index, m.index + m[0].length, fold(m[0])]);
  return out;
}
function wildcardForm(s) {
  const f = fold(s.replace(/^=+/, "")).replace(/[^\p{L}\p{N}*]/gu, "").replace(/\*+/g, "*");
  return f.replace(/\*/g, "").length >= 3 ? f : null;
}
// A query is a list of slots. A slot is a word, or {alternatives|separated|by|bars};
// any word may use * as a wildcard (vrijverklar*, *kinderen). Without ~N the slots must
// occur in this order, at most `near` words apart, as on the newspaper sites. With ~N,
// or when any slot is a {group}, they may occur in any order within N words.
function parseQuery(q) {
  const slots = [];
  let window = null;
  const alt = (part) => {
    const exact = part.startsWith("=");
    if (part.includes("*")) {
      const pat = wildcardForm(part);
      return pat ? [{ word: pat, exact: true, wildcard: true }] : [];
    }
    return tokenize(part.replace(/^=+/, "")).map((t) => ({ word: t[2], exact, wildcard: false }));
  };
  for (const chunk of q.split(/\s+/).filter(Boolean)) {
    const w = chunk.match(/^~(\d+)$/);
    if (w) { window = +w[1]; continue; }
    if (/^\{.*\}$/.test(chunk)) {
      // each alternative is one word: "slaven-kinderen" is read as slavenkinderen
      const alts = chunk.slice(1, -1).split("|").flatMap((p) => {
        const a = alt(p);
        return a.length > 1 ? [{ word: a.map((x) => x.word).join(""), exact: a[0].exact, wildcard: false }] : a;
      });
      if (alts.length) slots.push({ group: true, alts });
    } else {
      for (const a of alt(chunk)) slots.push({ group: false, alts: [a] });
    }
  }
  const s = state.meta ? state.meta.search : { default_window: 25, max_window: 60 };
  const unordered = window != null || slots.some((x) => x.group);
  return { slots, window: unordered ? Math.min(window || s.default_window, s.max_window) : null };
}
function fnv1a(s) {
  let h = 0x811c9dc5;
  for (const ch of s) {
    h ^= ch.codePointAt(0);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

// ------------------------------------------------------------------ thresholds
function binomCdf(k, n, p) {
  if (k < 0) return 0;
  if (k >= n) return 1;
  let term = Math.pow(1 - p, n), s = term;
  for (let i = 1; i <= k; i++) {
    term *= ((n - i + 1) / i) * (p / (1 - p));
    s += term;
  }
  return s;
}
function adaptiveK(ell, eps, eta) {
  for (let k = 0; k <= ell + 8; k++) if (binomCdf(k, ell, eps) >= 1 - eta) return k;
  return ell + 8;
}
function levenshtein(a, b, max) {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const v = Math.min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      cur.push(v);
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    prev = cur;
  }
  return prev[b.length];
}

// ------------------------------------------------------------------ data
async function fetchText(path) {
  const r = await fetch(DATA + path);
  if (!r.ok) throw new Error(`could not load ${path} (${r.status})`);
  return r.text();
}
async function loadVocab(n) {
  const lens = state.meta.vocab_lengths;
  if (n < 1 || n > state.meta.max_len || (lens && !lens.includes(n))) return { terms: [], arrays: [], df: new Map() };
  if (!state.vocab.has(n)) {
    const p = fetchText(`vocab/${n}.txt`).then((txt) => {
      const terms = [], df = new Map();
      for (const line of txt.split("\n")) {
        if (!line) continue;
        const [t, d] = line.split("\t");
        terms.push(t);
        df.set(t, +d);
      }
      return { terms, arrays: null, df };
    }).catch(() => ({ terms: [], arrays: [], df: new Map() }));
    state.vocab.set(n, p);
  }
  return state.vocab.get(n);
}
async function dfOf(form) {
  const v = await loadVocab([...form].length);
  return v.df.get(form) || 0;
}
async function postingsOf(form) {
  const h = fnv1a(form) % state.meta.post_shards;
  if (!state.postShards.has(h)) {
    state.postShards.set(h, fetchText(`post/${h}.txt`).then((txt) => {
      const m = new Map();
      for (const line of txt.split("\n")) {
        const tab = line.indexOf("\t");
        if (tab > 0) m.set(line.slice(0, tab), line.slice(tab + 1));
      }
      return m;
    }));
  }
  const raw = (await state.postShards.get(h)).get(form);
  if (!raw) return [];
  const out = [];
  let pid = 0;
  for (const part of raw.split(";")) {
    const [d, ps] = part.split(":");
    pid += parseInt(d, 36);
    out.push([pid, ps.split(",").map((x) => parseInt(x, 36))]);
  }
  return out;
}
async function passage(pid) {
  const n = Math.floor(pid / state.meta.text_shard);
  if (!state.textShards.has(n)) {
    state.textShards.set(n, fetch(`${DATA}text/${n}.json`).then((r) => r.json()).then((items) => {
      const m = new Map();
      for (const [p, page, text] of items) m.set(p, { page, text });
      return m;
    }));
  }
  return (await state.textShards.get(n)).get(pid);
}
function issueOf(pid) {
  const is = state.meta.issues;
  let lo = 0, hi = is.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (is[mid][0] <= pid) lo = mid; else hi = mid - 1;
  }
  const [first, last, date, paper, stratum, pages, , title] = is[lo];
  return { first, last, nPassages: last - first + 1, date, year: +date.slice(0, 4),
           paper: state.meta.papers[paper], paperIdx: paper, stratum, pages, title };
}
// Passages overlap; a match belongs to the passage whose first `step` words hold its start.
function ownedWords(pid, info) {
  return pid === info.last ? Infinity : state.meta.chunk.step;
}

// ------------------------------------------------------------------ search
async function expand(word, forceExact, opts) {
  const ell = [...word].length;
  const exact = forceExact || opts.exact || ell < opts.minFuzzy || opts.exactTerms.has(word);
  const k = exact ? 0 : adaptiveK(ell, opts.epsilon, opts.eta);
  const found = new Map();
  if (k === 0) {
    found.set(word, 0);
  } else {
    const w = [...word];
    for (let n = Math.max(ell - k, 1); n <= ell + k; n++) {
      const v = await loadVocab(n);
      if (!v.arrays) v.arrays = v.terms.map((t) => [...t]);
      for (let i = 0; i < v.terms.length; i++) {
        const d = levenshtein(w, v.arrays[i], k);
        if (d <= k) found.set(v.terms[i], d);
      }
    }
  }
  const variants = [];
  for (const [form, d] of found) {
    const df = await dfOf(form);
    if (df > 0) variants.push({ form, d, df });
  }
  variants.sort((a, b) => a.d - b.d || b.df - a.df || (a.form < b.form ? -1 : 1));
  // A misreading is much rarer than the word itself; a frequent "variant" is another
  // word (kinderen -> anderen, minderen). Such forms start unticked but stay listed.
  const own = variants.find((v) => v.d === 0);
  if (own) for (const v of variants) v.suspect = v.d > 0 && v.df > opts.suspectShare * own.df;
  return { word, k, exact, variants };
}
async function expandWildcard(pat, opts) {
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp("^" + pat.split("*").map(esc).join(".*") + "$", "u");
  const min = pat.replace(/\*/g, "").length;
  const vocabs = await Promise.all(state.meta.vocab_lengths.filter((n) => n >= min).map(loadVocab));
  const variants = [];
  for (const v of vocabs) for (const t of v.terms) if (re.test(t)) variants.push({ form: t, d: 0, df: v.df.get(t) });
  variants.sort((a, b) => b.df - a.df || (a.form < b.form ? -1 : 1));
  const cap = opts.maxWildcard;
  return { word: pat, k: 0, exact: true, wildcard: true,
           capped: variants.length > cap ? variants.length : 0, variants: variants.slice(0, cap) };
}

function passesFilters(info, f) {
  if (f.stratum && info.stratum !== f.stratum) return false;
  if (f.from && info.year < f.from) return false;
  if (f.to && info.year > f.to) return false;
  if (f.paper !== "" && f.paper != null && info.paperIdx !== +f.paper) return false;
  return true;
}

// In order, each word at most `near` words after the previous one.
function chainMatches(lists, near) {
  const matches = [];
  for (const first of lists[0]) {
    const chain = [first];
    for (const l of lists.slice(1)) {
      const prev = chain[chain.length - 1][0];
      const nxt = l.find((o) => o[0] > prev && o[0] <= prev + near + 1);
      if (!nxt) break;
      chain.push(nxt);
    }
    if (chain.length !== lists.length) continue;
    matches.push({ start: chain[0][0], end: chain[chain.length - 1][0], positions: chain.map((o) => o[0]),
                   forms: chain.map((o) => o[1]), dist: chain.reduce((s, o) => s + o[2], 0) });
  }
  return matches;
}
// Any order, one word of every slot within `win` words; shortest windows win, no overlaps.
function windowMatches(lists, win) {
  const k = lists.length;
  const all = [];
  lists.forEach((l, s) => { for (const o of l) all.push([o[0], s, o[1], o[2]]); });
  all.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cands = [];
  for (let i = 0; i < all.length; i++) {
    const pick = new Array(k).fill(null);
    let got = 0;
    for (let j = i; j < all.length && all[j][0] - all[i][0] <= win; j++) {
      if (pick[all[j][1]]) continue;
      pick[all[j][1]] = all[j];
      if (++got === k) { cands.push({ start: all[i][0], end: all[j][0], pick }); break; }
    }
  }
  cands.sort((a, b) => (a.end - a.start) - (b.end - b.start) || a.start - b.start);
  const kept = [];
  for (const c of cands) {
    if (kept.some((m) => c.start <= m.end && c.end >= m.start)) continue;
    kept.push({ start: c.start, end: c.end, positions: [...new Set(c.pick.map((o) => o[0]))].sort((a, b) => a - b),
                forms: c.pick.map((o) => o[2]), dist: c.pick.reduce((s, o) => s + o[3], 0) });
  }
  return kept;
}

async function searchOne(query, opts, filters) {
  const { slots, window } = parseQuery(query);
  if (!slots.length) return { hits: [], expansions: [] };
  const expansions = [];
  const active = [];
  for (const slot of slots) {
    const vs = new Map();
    for (const a of slot.alts) {
      const e = a.wildcard ? await expandWildcard(a.word, opts) : await expand(a.word, a.exact, opts);
      expansions.push(e);
      if (opts.autoExclude) for (const v of e.variants) if (v.suspect) state.excluded.add(v.form);
      for (const v of e.variants) {
        if (state.excluded.has(v.form)) continue;
        if (!vs.has(v.form) || vs.get(v.form).d > v.d) vs.set(v.form, v);
      }
    }
    active.push([...vs.values()]);
  }
  if (active.some((a) => !a.length)) return { hits: [], expansions };

  // per slot: pid -> [[pos, form, d], ...]
  const occ = [];
  for (const vs of active) {
    const m = new Map();
    const lists = await Promise.all(vs.map((v) => postingsOf(v.form)));
    vs.forEach((v, i) => {
      for (const [pid, ps] of lists[i]) {
        if (!m.has(pid)) m.set(pid, []);
        for (const p of ps) m.get(pid).push([p, v.form, v.d]);
      }
    });
    occ.push(m);
  }
  const lengths = slots.reduce((s, sl) => s + Math.max(...sl.alts.map((a) => [...a.word.replace(/\*/g, "")].length)), 0);
  const N = state.meta.n_passages;
  const dfs = new Map(active.flat().map((v) => [v.form, v.df]));
  const idf = (f) => Math.log((N + 1) / ((dfs.get(f) || 0) + 0.5));
  const smallest = occ.reduce((a, b) => (a.size <= b.size ? a : b));

  const hits = [];
  for (const pid of smallest.keys()) {
    if (!occ.every((m) => m.has(pid))) continue;
    const info = issueOf(pid);
    if (!passesFilters(info, filters)) continue;
    const lists = occ.map((m) => m.get(pid).slice().sort((a, b) => a[0] - b[0]));
    const own = ownedWords(pid, info);
    const matches = (window == null ? chainMatches(lists, opts.near) : windowMatches(lists, window))
      .filter((m) => m.start < own)
      .map((m) => ({ ...m, query, score: (1 - m.dist / Math.max(lengths, 1)) * m.forms.reduce((s, f) => s + idf(f), 0) }));
    if (!matches.length) continue;
    const best = Math.max(...matches.map((m) => m.score));
    matches.sort((a, b) => a.dist - b.dist || a.start - b.start);
    hits.push({ pid, info, matches, score: best * (1 + 0.1 * (Math.min(matches.length, 10) - 1)) });
  }
  return { hits, expansions };
}

async function loadConcept(key) {
  const def = state.meta.concepts.find((c) => c.key === key);
  if (!def) throw new Error(`unknown concept ${key}`);
  if (!state.concepts.has(key)) state.concepts.set(key, fetch(DATA + def.file).then((r) => r.json()));
  return { def, rows: await state.concepts.get(key) };
}
async function searchConcept(key, filters) {
  const { def, rows } = await loadConcept(key);
  const hits = [];
  for (const [pid, score, tier, ms] of rows) {
    if (state.tiersOff.has(tier)) continue;
    const info = issueOf(pid);
    if (!passesFilters(info, filters)) continue;
    hits.push({ pid, info, score, tier,
                matches: ms.map(([start, end, positions]) => ({ start, end, positions, forms: [], dist: 0, tier })) });
  }
  return { hits, expansions: [], concept: def };
}

function cmpDate(a, b) {
  if (a.info.date !== b.info.date) return a.info.date < b.info.date ? -1 : 1;
  return a.pid - b.pid;
}

function applySort(hits) {
  const mode = ($("sort") && $("sort").value) || "oldest";
  if (mode === "newest") hits.sort((a, b) => -cmpDate(a, b) || b.score - a.score);
  else if (mode === "score") hits.sort((a, b) => b.score - a.score || cmpDate(a, b));
  else hits.sort((a, b) => cmpDate(a, b) || b.score - a.score);
  return hits;
}

function currentSort() {
  return ($("sort") && $("sort").value) || "oldest";
}

function sortLabel() {
  return currentSort() === "newest" ? "newest first" : currentSort() === "score" ? "best match first" : "oldest first";
}

function updateSortHint() {
  const el = $("summary-text") && $("summary-text").querySelector(".hint");
  if (el) el.textContent = " · " + sortLabel();
}

// ------------------------------------------------------------------ rendering
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const prettyDate = (d) => `${+d.slice(8, 10)} ${MONTHS[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const periodIndex = (s) => state.meta.periods.findIndex((p) => p.key === s);
const periodLabel = (s) => { const p = state.meta.periods[periodIndex(s)]; return p ? p.label : ""; };
const tierLabel = (t) => {
  const c = state.meta.concepts.flatMap((x) => x.tiers).find((x) => x.key === t);
  return c ? c.label : t;
};
function queryLabel(q = state.lastQuery) {
  if (!q) return "";
  if (q.concept) return `(${state.meta.concepts.find((c) => c.key === q.concept).label} rules)`;
  return q.query;
}

function status(msg, isError = false) {
  $("status").textContent = msg;
  $("status").className = isError ? "error" : "";
}

function pageLink(info) {
  const id = info.pages && info.pages[0];
  return id ? state.meta.page_url.replaceAll("{id}", id) : null;
}

function highlighted(text, spans, from, to) {
  let out = "", pos = from;
  for (const [a, b] of spans) {
    if (b <= from || a >= to || a < pos) continue;
    out += esc(text.slice(pos, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>";
    pos = b;
  }
  return out + esc(text.slice(pos, to));
}
// Phrase matches are marked as one stretch; window matches word by word.
function matchSpans(toks, matches) {
  const spans = [];
  for (const m of matches) {
    if (!toks[m.start] || !toks[m.end]) continue;
    if (m.positions && m.end - m.start > 3) {
      for (const i of m.positions) if (toks[i]) spans.push([toks[i][0], toks[i][1]]);
    } else spans.push([toks[m.start][0], toks[m.end][1]]);
  }
  return spans.sort((a, b) => a[0] - b[0]);
}
function foundWords(toks, matches) {
  const out = new Set();
  for (const m of matches) {
    const words = m.forms && m.forms.length ? m.forms : (m.positions || []).map((i) => toks[i] && toks[i][2]).filter(Boolean);
    out.add(words.join(" "));
  }
  return [...out];
}

async function renderHit(h, rank) {
  const p = await passage(h.pid);
  const toks = tokenize(p.text);
  const spans = matchSpans(toks, h.matches);
  const m0 = h.matches.find((m) => toks[m.start] && toks[m.end]);
  const first = m0 ? toks[m0.start][0] : 0, last = m0 ? toks[m0.end][1] : 0;
  const a = Math.max(0, first - 220), b = Math.min(p.text.length, Math.max(first + 380, last + 160));
  const snippet = (a > 0 ? "… " : "") + highlighted(p.text, spans, a, b) + (b < p.text.length ? " …" : "");
  h.snippet = p.text.slice(a, b);
  h.page = p.page;
  h.found = foundWords(toks, h.matches);

  const link = pageLink(h.info);
  const rev = reviews()[h.pid] || {};
  const ocrFix = ocrCorrection(h.pid);
  const div = document.createElement("article");
  div.className = "hit noclip";
  div.dataset.pid = h.pid;
  div.innerHTML = `
    <div>
      <h3>${rank}. ${prettyDate(h.info.date)} — <i>${esc(h.info.title)}</i>
        ${h.tier ? `<span class="tier tier-${esc(h.tier)}" title="score ${h.score.toFixed(2)}">${esc(tierLabel(h.tier))}</span>` : ""}</h3>
      <div class="where">Passage ${p.page} of ${h.info.nPassages} · ${esc(periodLabel(h.info.stratum))}
        · found: ${h.found.slice(0, 4).map(esc).join(", ")}</div>
      <div class="text">${snippet}</div>
      <div class="links">
        ${link ? `<a href="${link}" target="_blank" rel="noopener">${esc(state.meta.page_link_text)} ↗</a>` : ""}
        <span class="hint">document ${esc(h.info.pages[0])}</span>
        <button type="button" class="toggle-full">Show whole passage</button>
        <button type="button" class="edit-ocr">${ocrFix ? "Edit OCR correction" : "Correct OCR"}</button>
      </div>
      <div class="full" hidden>${highlighted(p.text, spans, 0, p.text.length)}</div>
      <div class="ocr-saved"${ocrFix ? "" : " hidden"}><b>Your correction:</b> <span class="ocr-saved-text"></span></div>
      <div class="ocr-editor" hidden>
        <p class="hint">Look at the scan (${esc(state.meta.library)}) and type what the print actually says.
          The computer's reading is kept for comparison. This does not change the search index.</p>
        <textarea rows="7" spellcheck="true"></textarea>
        <div class="ocr-actions">
          <button type="button" class="save-ocr primary">Save correction</button>
          <button type="button" class="revert-ocr">Revert to computer OCR</button>
        </div>
      </div>
      <div class="review">Is this about freeing children born to enslaved mothers?
        ${LABELS.map(([key, text]) => `<button type="button" data-label="${key}" aria-pressed="${rev.label === key}">${text}</button>`).join("")}
        <input type="text" placeholder="Note (optional)" value="${esc(rev.note || "")}">
      </div>
    </div>`;
  div.querySelector(".toggle-full").onclick = (e) => {
    const f = div.querySelector(".full");
    f.hidden = !f.hidden;
    e.target.textContent = f.hidden ? "Show whole passage" : "Hide whole passage";
  };
  for (const btn of div.querySelectorAll(".review button")) {
    btn.onclick = () => {
      const label = btn.dataset.label;
      const cur = reviews()[h.pid];
      const next = cur && cur.label === label ? null : label;
      saveReview(h, next, div.querySelector(".review input").value);
      for (const x of div.querySelectorAll(".review button")) x.setAttribute("aria-pressed", x.dataset.label === next);
    };
  }
  div.querySelector(".review input").onchange = (e) => {
    const cur = reviews()[h.pid];
    saveReview(h, cur ? cur.label : null, e.target.value);
  };
  const editor = div.querySelector(".ocr-editor");
  const area = editor.querySelector("textarea");
  const savedBox = div.querySelector(".ocr-saved");
  const savedText = div.querySelector(".ocr-saved-text");
  const editBtn = div.querySelector(".edit-ocr");
  area.value = ocrFix ? ocrFix.corrected : p.text;
  if (ocrFix) savedText.textContent = ocrFix.corrected;
  const paintOcr = (fix) => {
    editBtn.textContent = fix ? "Edit OCR correction" : "Correct OCR";
    savedBox.hidden = !fix;
    savedText.textContent = fix ? fix.corrected : "";
    area.value = fix ? fix.corrected : p.text;
  };
  editBtn.onclick = () => {
    editor.hidden = !editor.hidden;
    if (!editor.hidden) area.focus();
  };
  editor.querySelector(".save-ocr").onclick = () => {
    const fix = saveOcrCorrection(h, p.text, area.value);
    paintOcr(fix);
    editor.hidden = true;
    status(fix ? "OCR correction saved in this browser." : "Reverted to the computer's OCR.");
  };
  editor.querySelector(".revert-ocr").onclick = () => {
    area.value = p.text;
    paintOcr(saveOcrCorrection(h, p.text, p.text));
    editor.hidden = true;
    status("Reverted to the computer's OCR.");
  };
  return div;
}

async function showMore() {
  const batch = state.hits.slice(state.shown, state.shown + PAGE_SIZE);
  $("more").disabled = true;
  const els = await Promise.all(batch.map((h, i) => renderHit(h, state.shown + i + 1)));
  for (const el of els) $("results").appendChild(el);
  state.shown += batch.length;
  $("more").disabled = false;
  $("more").hidden = state.shown >= state.hits.length;
}

function renderSummary(hits, expansions, concept) {
  const docs = new Set(hits.map((h) => h.info.first));
  $("summary-text").innerHTML = hits.length
    ? `<b>${hits.length.toLocaleString()}</b> passages in <b>${docs.size.toLocaleString()}</b> documents
       <span class="hint"> · ${sortLabel()}</span>`
    : "Nothing found. Try another spelling, or remove a filter.";
  const byYear = new Map();
  for (const h of hits) byYear.set(h.info.year, (byYear.get(h.info.year) || 0) + 1);
  const years = [...byYear.keys()].sort();
  const max = Math.max(1, ...byYear.values());
  const yearPeriod = (y) => state.meta.periods.findIndex((p) => p.from <= y && y <= p.to);
  $("years").innerHTML = years.length > 1 ? years.map((y) =>
    `<div class="bar p${yearPeriod(y) + 1}" style="height:${(byYear.get(y) / max) * 100}%" title="${y}: ${byYear.get(y)} passages"><span>${y}</span></div>`).join("") : "";

  $("concept-box").hidden = !concept;
  $("variants-box").hidden = !!concept;
  if (concept) {
    $("concept-rules").innerHTML =
      `<p><b>${esc(concept.title)}.</b> A passage is listed when a word of each kind occurs close together:</p><ul>` +
      Object.entries(concept.slots).map(([k, v]) => `<li><b>${esc(k)}</b>: ${esc(v)}</li>`).join("") + "</ul>";
    $("tiers").innerHTML = concept.tiers.map((t) =>
      `<label title="${esc(t.rule)}"><input type="checkbox" data-tier="${esc(t.key)}" ${state.tiersOff.has(t.key) ? "" : "checked"}>
        <span class="tier tier-${esc(t.key)}">${esc(t.label)}</span> <span class="hint">${esc(t.rule)}</span></label>`).join("");
    for (const box of $("tiers").querySelectorAll("input")) {
      box.onchange = () => {
        if (box.checked) state.tiersOff.delete(box.dataset.tier); else state.tiersOff.add(box.dataset.tier);
        run("", { concept: concept.key, keepExcluded: true });
      };
    }
  } else {
    const seen = new Set();
    let html = "";
    for (const e of expansions) {
      if (seen.has(e.word)) continue;
      seen.add(e.word);
      const note = e.wildcard ? (e.capped ? ` (wildcard; the ${e.variants.length} most frequent of ${e.capped} forms)` : " (wildcard)")
                              : e.exact ? " (exact spelling)" : "";
      html += `<div class="word">${esc(e.word)}${note}</div>`;
      html += e.variants.length ? e.variants.map((v) =>
        `<label${v.suspect ? ' title="Much more common than the word you typed, so probably a different word"' : ""}>
          <input type="checkbox" data-form="${esc(v.form)}" ${state.excluded.has(v.form) ? "" : "checked"}> ${esc(v.form)}
          <span class="hint">${v.df.toLocaleString()}×${v.suspect ? ", probably another word" : ""}</span></label>`).join("")
        : `<span class="hint">not in these papers</span>`;
    }
    $("variants").innerHTML = html;
  }
  $("summary").hidden = false;
}

// ------------------------------------------------------------------ run
function currentFilters() {
  return {
    stratum: $("period").value,
    from: +$("from").value || null,
    to: +$("to").value || null,
    paper: $("paper").value,
  };
}
function currentOpts() {
  const s = state.meta.search;
  return {
    exact: $("exact").checked,
    epsilon: s.epsilon, eta: s.eta, minFuzzy: s.min_fuzzy_len, near: s.near,
    maxWildcard: s.max_wildcard_forms,
    suspectShare: s.suspect_share,
    autoExclude: !(state.lastQuery && state.lastQuery.keepExcluded),
    exactTerms: new Set(state.meta.exact_only),
  };
}

async function run(query, { concept = null, keepExcluded = false } = {}) {
  if (!keepExcluded) state.excluded.clear();
  state.lastQuery = { query, concept, keepExcluded };
  if (!concept && !parseQuery(query).slots.length) { status("Type a word to search for."); return; }
  $("results").innerHTML = "";
  $("more").hidden = true;
  $("summary").hidden = true;
  status("Searching…");
  const t0 = performance.now();
  try {
    const r = concept ? await searchConcept(concept, currentFilters())
                      : await searchOne(query, currentOpts(), currentFilters());
    state.hits = applySort(r.hits);
    state.expansions = r.expansions;
    state.shown = 0;
    renderSummary(r.hits, r.expansions, r.concept);
    status(`Search took ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
    writeHash();
    await showMore();
  } catch (err) {
    console.error(err);
    status("Something went wrong while searching: " + err.message, true);
  }
}

function writeHash() {
  const f = currentFilters();
  const p = new URLSearchParams();
  if (state.lastQuery.concept) p.set("concept", state.lastQuery.concept); else p.set("q", state.lastQuery.query);
  if (state.tiersOff.size) p.set("hide", [...state.tiersOff].join(","));
  if (f.stratum) p.set("period", f.stratum);
  if (f.from) p.set("from", f.from);
  if (f.to) p.set("to", f.to);
  if (f.paper !== "") p.set("source", f.paper);
  if ($("exact").checked) p.set("exact", "1");
  if (currentSort() !== "oldest") p.set("sort", currentSort());
  history.replaceState(null, "", "#" + p.toString());
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  $("q").value = p.get("q") || "";
  $("period").value = p.get("period") || "";
  $("from").value = p.get("from") || "";
  $("to").value = p.get("to") || "";
  $("paper").value = p.get("source") || "";
  $("exact").checked = p.get("exact") === "1";
  $("sort").value = ["oldest", "newest", "score"].includes(p.get("sort")) ? p.get("sort") : "oldest";
  state.tiersOff = new Set((p.get("hide") || "").split(",").filter(Boolean));
  if (p.get("concept")) return run("", { concept: p.get("concept"), keepExcluded: true });
  if (p.get("q")) return run(p.get("q"));
}

// ------------------------------------------------------------------ reviews & downloads
function reviews() {
  try { return JSON.parse(localStorage.getItem(reviewKey())) || {}; } catch { return {}; }
}
function ocrCorrections() {
  try { return JSON.parse(localStorage.getItem(ocrKey())) || {}; } catch { return {}; }
}
function ocrCorrection(pid) {
  return ocrCorrections()[pid] || null;
}
function hitRecord(h) {
  return {
    query: queryLabel(), date: h.info.date, source: h.info.paper, document: h.info.title,
    doc_id: h.info.pages[0], passage: h.page, tier: h.tier || "", link: pageLink(h.info) || "",
  };
}
function saveOcrCorrection(h, original, corrected) {
  const all = ocrCorrections();
  const next = (corrected || "").replace(/\r\n/g, "\n");
  const orig = (original || "").replace(/\r\n/g, "\n");
  if (!next.trim() || next === orig) {
    delete all[h.pid];
    localStorage.setItem(ocrKey(), JSON.stringify(all));
    updateReviewCount();
    return null;
  }
  all[h.pid] = { original: orig, corrected: next, time: new Date().toISOString(), ...hitRecord(h) };
  localStorage.setItem(ocrKey(), JSON.stringify(all));
  updateReviewCount();
  return all[h.pid];
}
function saveReview(h, label, note) {
  const all = reviews();
  if (!label && !note) delete all[h.pid];
  else all[h.pid] = { label, note, time: new Date().toISOString(), text: h.snippet || "", ...hitRecord(h) };
  localStorage.setItem(reviewKey(), JSON.stringify(all));
  updateReviewCount();
}
function updateReviewCount() {
  const all = Object.values(reviews());
  const n = (l) => all.filter((r) => r.label === l).length;
  const nOcr = Object.keys(ocrCorrections()).length;
  const checks = all.length
    ? `${all.length} checked: ` + LABELS.map(([k, text]) => `${n(k)} ${text.toLowerCase()}`).join(", ")
    : "No passages checked yet";
  $("review-count").textContent = nOcr ? `${checks}; ${nOcr} OCR correction${nOcr === 1 ? "" : "s"}.` : `${checks}.`;
}
function csv(rows) {
  const cell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n;]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return "﻿" + rows.map((r) => r.map(cell).join(",")).join("\r\n");
}
function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function matchedSurface(text, matches) {
  const toks = tokenize(text);
  return [...new Set(matchSpans(toks, matches).map(([a, b]) => text.slice(a, b)))].join(" | ");
}

async function exportSearchList() {
  if (!state.hits.length) { status("Search first, then download."); return; }
  const btn = $("download-results");
  btn.disabled = true;
  status("Preparing the spreadsheet (including OCR text)…");
  try {
    const rows = [["rank", "score", "tier", "date", "source", "document", "passage", "ocr text", "corrected ocr",
                   "matched ocr", "words found", "passage id", "document id", "period", "link", "label", "note"]];
    const revs = reviews();
    for (let i = 0; i < state.hits.length; i++) {
      const h = state.hits[i];
      const p = await passage(h.pid);
      const fix = ocrCorrection(h.pid);
      const r = revs[h.pid] || {};
      rows.push([
        i + 1, h.score.toFixed(3), h.tier || "", h.info.date, h.info.paper, h.info.title, p.page, p.text,
        fix ? fix.corrected : "", matchedSurface(p.text, h.matches), foundWords(tokenize(p.text), h.matches).join("; "),
        h.pid, h.info.pages[0], periodLabel(h.info.stratum), pageLink(h.info) || "", r.label || "", r.note || "",
      ]);
      if (i % 200 === 199) status(`Preparing the spreadsheet… ${i + 1}/${state.hits.length}`);
    }
    download("search-results.csv", csv(rows));
    status(`Saved ${state.hits.length} passages with OCR text.`);
  } catch (err) {
    console.error(err);
    status("Could not prepare the download: " + err.message, true);
  } finally {
    btn.disabled = false;
  }
}

function init() {
  $("search-form").onsubmit = (e) => { e.preventDefault(); run($("q").value); };
  for (const chip of document.querySelectorAll(".chip[data-q]")) chip.onclick = () => { $("q").value = chip.dataset.q; run(chip.dataset.q); };
  for (const chip of document.querySelectorAll(".chip[data-concept]")) {
    chip.onclick = () => { $("q").value = ""; state.tiersOff.clear(); run("", { concept: chip.dataset.concept }); };
  }
  $("sort").onchange = async () => {
    writeHash();
    updateSortHint();
    if (!state.hits.length) return;
    applySort(state.hits);
    state.shown = 0;
    $("results").innerHTML = "";
    $("more").hidden = true;
    await showMore();
  };
  $("more").onclick = showMore;
  $("rerun").onclick = () => {
    state.excluded = new Set([...document.querySelectorAll("#variants input:not(:checked)")].map((x) => x.dataset.form));
    run(state.lastQuery.query, { keepExcluded: true });
  };
  $("copy-link").onclick = async () => {
    try { await navigator.clipboard.writeText(location.href); status("Link copied. Paste it into an email to share this search."); }
    catch { status("Copy this link from the address bar: " + location.href); }
  };
  $("download-results").onclick = () => exportSearchList();
  $("download-reviews").onclick = () => {
    const who = $("reviewer").value;
    const labels = reviews();
    const fixes = ocrCorrections();
    const code = Object.fromEntries(LABELS.map(([k, , c]) => [k, c]));
    const pids = [...new Set([...Object.keys(labels), ...Object.keys(fixes)])];
    const rows = [["reviewer", "passage id", "about free womb", "label code (2/1/0)", "note", "date", "source", "document",
                   "passage", "tier", "search", "ocr text", "corrected ocr", "link", "checked at"]];
    for (const pid of pids) {
      const r = labels[pid] || {};
      const f = fixes[pid] || {};
      const x = r.date ? r : f;
      rows.push([who, pid, r.label || "", r.label ? code[r.label] : "", r.note || "", x.date || "", x.source || "",
                 x.document || "", x.passage || "", x.tier || "", x.query || "", f.original || r.text || "",
                 f.corrected || "", x.link || "", r.time || f.time || ""]);
    }
    download(`checks-${(who || "reviewer").replace(/\W+/g, "_")}-${new Date().toISOString().slice(0, 10)}.csv`, csv(rows));
  };
  $("clear-reviews").onclick = () => {
    if (confirm("Delete all your checks and OCR corrections from this browser? Download them first if you need them.")) {
      localStorage.removeItem(reviewKey());
      localStorage.removeItem(ocrKey());
      updateReviewCount();
      for (const b of document.querySelectorAll(".review button")) b.setAttribute("aria-pressed", "false");
      for (const box of document.querySelectorAll(".ocr-saved")) box.hidden = true;
      for (const btn of document.querySelectorAll(".edit-ocr")) btn.textContent = "Correct OCR";
    }
  };
}

(async function main() {
  init();
  try {
    state.meta = await (await fetch(DATA + "meta.json")).json();
  } catch (err) {
    status("Could not load the index. If you opened this file directly from your computer, " +
           "use the website address instead.", true);
    return;
  }
  const m = state.meta;
  $("reviewer").value = localStorage.getItem(reviewKey() + "-name") || "";
  $("reviewer").onchange = (e) => localStorage.setItem(reviewKey() + "-name", e.target.value);
  updateReviewCount();
  const years = m.issues.map((i) => i[2].slice(0, 4)).sort();
  $("corpus-size").textContent =
    `${m.n_issues.toLocaleString()} documents (${years[0]}–${years[years.length - 1]}, ${m.n_passages.toLocaleString()} passages)`;
  $("built").textContent = `Index built ${m.built}.`;
  m.papers.forEach((p, i) => $("paper").insertAdjacentHTML("beforeend", `<option value="${i}">${esc(p)}</option>`));
  m.periods.forEach((p) => $("period").insertAdjacentHTML("beforeend", `<option value="${esc(p.key)}">${esc(p.label)}</option>`));
  for (const c of m.concepts) {
    const chip = document.querySelector(`.chip[data-concept="${c.key}"]`);
    if (chip) chip.title = `${c.title}: ${c.n.toLocaleString()} passages found by the stage-1 rules`;
  }
  window.addEventListener("hashchange", readHash);
  await readHash();
})();
