// rizzo-pii live demo: everything runs client-side.
// Tokenizer: transformers.js · Model: onnxruntime-web (int8 ONNX of rizzoaiacademy/rizzo-pii-0.3B)
// Regex/checksum net, merge, placeholders, restore: the project's Python, run in Pyodide.
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";

const MODEL_REPO = "r69shabh/rizzo-pii-0.3B-onnx-web";
const MODEL_URL = `https://huggingface.co/${MODEL_REPO}/resolve/main/onnx/model_quantized.onnx`;
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

const $ = (id) => document.getElementById(id);
const status = (html, cls = "") => { const s = $("status"); s.className = "status " + cls; s.innerHTML = html; };

const TAGS = ["FULLNAME","AGE","GENDER","DATE","TIME","STREET","BUILDINGNUM","ZIPCODE","CITY","PROVINCE","EMAIL","TELEPHONENUM","CF","PIVA","ID_DOC","IBAN","CREDITCARDNUMBER","AMOUNT","TARGA","ORG","DOCID","CATASTO","URL","IPADDR"];
const PAL = ["#ddd6fe","#fecaca","#fed7aa","#fde68a","#d9f99d","#bbf7d0","#a7f3d0","#99f6e4","#a5f3fc","#bae6fd","#bfdbfe","#c7d2fe","#f5d0fe","#fbcfe8","#fecdd3","#e9d5ff","#fef08a","#bef264","#86efac","#5eead4","#7dd3fc","#a5b4fc","#f0abfc","#fda4af"];
const COLOR = Object.fromEntries(TAGS.map((t, i) => [t, PAL[i]]));

const EX = [
`CONTRATTO DI LOCAZIONE AD USO ABITATIVO

Tra il Sig. Mario Rossi, nato a Milano il 12/06/1985, codice fiscale RSSMRA85H12F205Y, residente in Via Garibaldi 24, 20121 Milano (MI), di seguito "Locatore", e la Sig.ra Giulia Bianchi, codice fiscale BNCGLI90A41H501E, tel. +39 333 1234567, email giulia.bianchi@pec.it, di seguito "Conduttore", si conviene quanto segue: il canone annuo è fissato in € 12.500,00, da versare mediante bonifico sull'IBAN IT60 X054 2811 1010 0000 0123 456 intestato al Locatore. L'immobile è censito al Catasto Fabbricati del Comune di Milano al Foglio 12, part. 345, sub. 6.`,
`TRIBUNALE DI ROMA – Sezione Lavoro
R.G. n. 1234/2024

Il Giudice, dott. Marco Conti, sentito l'Avv. Anna Ferrari del Foro di Roma, difensore della società Edilnord S.r.l. (P.IVA 01234567897, con sede in Piazza Navona 3, 00186 Roma), all'udienza del 14 marzo 2025 alle ore 10:30, rinvia la causa promossa dal ricorrente Luca Verdi (C.F. VRDLCU78C15L219W), di anni 46, al 3 ottobre 2025.`,
`Oggetto: richiesta rimborso sinistro

Gentile Studio Legale Ferrari & Associati,
vi scrivo in merito al sinistro del 02/02/2025 che ha coinvolto la mia auto targata AB 123 CD. Allego copia della carta d'identità n. CA12345AB. Per il rimborso di € 1.840,50 potete usare la carta 4111 1111 1111 1111 oppure contattarmi al 347-555-0192.

Cordiali saluti,
Anna Ferrari
Via Roma 15, 10121 Torino (TO)`];
const REPLY_EX = `Ecco una sintesi del contratto:

- **Locatore:** [FULLNAME_1] ([CF_1]), residente in [STREET_1] [BUILDINGNUM_1], [CITY_1].
- **Conduttore:** [FULLNAME_2], contattabile al [TELEPHONENUM_1].
- **Canone:** [AMOUNT_1] annui, da versare sull'IBAN [IBAN_1].

Consiglio di inviare la bozza a FULLNAME_2 all'indirizzo [EMAIL_1].`;

let tok, sess, py, MAP = {};

// ---------- UI wiring ----------
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => showTab(b.dataset.t));
function showTab(t) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x.dataset.t === t));
  $("p1").classList.toggle("hidden", t !== "p1");
  $("p2").classList.toggle("hidden", t !== "p2");
}
document.querySelectorAll("[data-ex]").forEach((b) => b.onclick = () => { $("inp").value = EX[+b.dataset.ex]; });
$("tags").innerHTML = TAGS.map((t) => `<label><input type="checkbox" value="${t}"> ${t}</label>`).join("");
$("inp").value = EX[0];
$("rin").value = REPLY_EX;
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const copy = (id) => navigator.clipboard.writeText($(id).value);
$("copy").onclick = () => copy("anon");
$("rcopy").onclick = () => copy("rout");
$("to2").onclick = () => showTab("p2");
$("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(MAP, null, 2)], { type: "application/json" }));
  a.download = "dizionario_anonimizzazione.json"; a.click();
};

// ---------- loading ----------
async function fetchWithProgress(url) {
  const cache = await caches.open("rizzo-pii-demo-v1");
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model download failed (${r.status})`);
  const total = +r.headers.get("content-length") || 193144387;
  const reader = r.body.getReader(); const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    parts.push(value); got += value.length;
    const p = document.getElementById("prog"); if (p) p.style.width = Math.min(100, (got / total) * 100) + "%";
  }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  try { await cache.put(url, new Response(buf)); } catch (_) {}
  return buf;
}

async function loadPython() {
  const { loadPyodide } = await import(PYODIDE + "pyodide.mjs");
  const p = await loadPyodide({ indexURL: PYODIDE });
  for (const f of ["detectors.py", "core.py"]) {
    p.FS.writeFile(f, await (await fetch(f)).text());
  }
  p.runPython("import sys; sys.path.insert(0, '.'); import core");
  return p;
}

async function retry(fn, label, n = 4) {
  for (let i = 1; ; i++) {
    try { return await fn(); }
    catch (e) {
      console.warn(label, "attempt", i, e);
      if (i >= n) throw new Error(`${label}: ${e.message || e}`);
      await new Promise((r) => setTimeout(r, 1500 * i));
    }
  }
}

async function init() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    ort.env.wasm.numThreads = 1;
    env.allowLocalModels = false;
    const [t, p, bytes, cfg] = await Promise.all([
      retry(() => AutoTokenizer.from_pretrained(MODEL_REPO), "tokenizer"),
      retry(loadPython, "python runtime"),
      retry(() => fetchWithProgress(MODEL_URL), "model"),
      retry(() => fetch(`https://huggingface.co/${MODEL_REPO}/resolve/main/config.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }), "config"),
    ]);
    tok = t; py = p;
    status("Model downloaded, starting the inference engine…");
    sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "basic" });
    py.globals.get("core").set_labels(JSON.stringify(cfg.id2label));
    status("✅ Ready. Everything runs locally in this tab: nothing you paste is uploaded.", "ok");
    $("go").disabled = false; $("rev").disabled = false;
    window.__ready = true;
  } catch (e) {
    console.error(e);
    status("⚠️ Could not load the model: " + esc(String(e.message || e)) + ". Please reload the page (desktop Chrome / Edge / Firefox work best).", "err");
  }
}

// ---------- inference ----------
async function runChunk(text) {
  const enc = tok(text);
  const ids = Array.from(enc.input_ids.data, Number);
  const n = ids.length;
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]),
    attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
  };
  const out = await sess.run(feeds);
  const lg = out.logits.data, L = out.logits.dims[2];
  const rows = [];
  for (let i = 0; i < n; i++) rows.push(Array.from(lg.subarray(i * L, (i + 1) * L)));
  return { tokens: tok.tokenize(text), logits: rows };
}

async function anonymize() {
  const text = $("inp").value.trim();
  if (!text) return;
  if (text.length > 30000) { status("This demo handles up to 30,000 characters at a time.", "err"); return; }
  $("go").disabled = true; const t0 = performance.now();
  try {
    const core = py.globals.get("core");
    const chunks = JSON.parse(core.chunks_json(text));
    const res = [];
    for (let i = 0; i < chunks.length; i++) {
      $("t").textContent = `chunk ${i + 1}/${chunks.length}…`;
      res.push(await runChunk(chunks[i]));
      await new Promise((r) => setTimeout(r, 0));
    }
    const excluded = [...document.querySelectorAll("#tags input:checked")].map((x) => x.value);
    const out = JSON.parse(core.analyze(text, JSON.stringify(res), JSON.stringify(excluded)));
    MAP = out.mapping;
    $("hl").classList.remove("ex");
    $("hl").innerHTML = out.segments.map((s) => s.label
      ? `<span class="ent${s.validated ? " v" : ""}" style="background:${COLOR[s.label] || "#eee"}" title="${s.ph}${s.validated ? " · checksum verified" : ""}">${esc(s.t)}<small>${s.label}</small></span>`
      : esc(s.t)).join("");
    $("stats").innerHTML = `<b>${out.n_entities}</b> entities (${out.n_unique} unique) · <b>${out.n_validated}</b> checksum-verified<br/>` +
      Object.entries(out.by_label).map(([k, v]) => `<code>${k}</code> ${v}`).join(" · ");
    $("anon").value = out.anonymized_text;
    $("map").textContent = JSON.stringify(MAP, null, 2);
    $("rmap").value = JSON.stringify(MAP, null, 2);
    $("t").textContent = `done in ${((performance.now() - t0) / 1000).toFixed(1)} s`;
    window.__last = out;
  } catch (e) {
    console.error(e); $("t").textContent = "error: " + (e.message || e);
  } finally { $("go").disabled = false; }
}

function restore() {
  try {
    JSON.parse($("rmap").value || "{}");
  } catch (_) { $("rout").value = "The dictionary is not valid JSON."; return; }
  $("rout").value = py.globals.get("core").restore($("rin").value, $("rmap").value);
}

$("go").onclick = anonymize;
$("rev").onclick = restore;
init();
