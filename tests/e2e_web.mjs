// End-to-end test of the phone web app using its synthetic demo engine.
// Builds a baseline (3 states x 3 takes), then runs healthy and faulty checks
// and prints each verdict. Exits non-zero if healthy checks mostly alarm or
// faults are mostly missed.
//
//   npm i playwright            (or use a preinstalled one)
//   node tests/e2e_web.mjs [--tfjs /path/to/tf.min.js] [--runs 3]
//
// --tfjs serves TensorFlow.js from a local file instead of the CDN (offline CI).
import { chromium } from "playwright";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import http from "node:http";

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const tfjs = opt("--tfjs", null), runs = +opt("--runs", 3);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
const errors = [];
page.on("pageerror", e => errors.push(String(e)));
if (tfjs) await page.route("https://cdn.jsdelivr.net/npm/@tensorflow/**", r => r.fulfill({ path: tfjs, contentType: "application/javascript" }));
await page.route("https://fonts.googleapis.com/**", r => r.fulfill({ body: "", contentType: "text/css" }));
// serve docs/ over http so the app can fetch its model files
const docs = path.join(here, "..", "docs");
const types = { ".html": "text/html", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const f = path.join(docs, decodeURIComponent(req.url.split("?")[0]).replace(/^\/$/, "/index.html"));
  if (!f.startsWith(docs) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "Content-Type": types[path.extname(f)] || "application/octet-stream" }); fs.createReadStream(f).pipe(res);
}).listen(0);
await page.goto(`http://127.0.0.1:${server.address().port}/index.html`);

const click = async sel => { await page.locator(sel).first().click(); await page.waitForTimeout(30); };
await click("[data-act=demo]");
for (let i = 0; i < 9; i++) await click("[data-act=demorec]");
await page.waitForSelector("text=Baseline saved", { timeout: 180000 });

async function check(fault) {
  await click("[data-act=check]");
  for (let s = 0; s < 3; s++) await click(`[data-act=demorec][data-fault="${fault}"]`);
  await page.waitForSelector(".verdict h1");
  const r = await page.evaluate(() => {
    const cars = JSON.parse(localStorage.getItem("engineear.cars"));
    const h = cars.find(c => c.id === "demo").history.at(-1);
    return { verdict: h.verdict, likely: (h.likely || []).map(c => `${c.cls} ${Math.round(100 * c.p)}%`).join(", "), states: Object.fromEntries(Object.entries(h.states).map(([s, v]) => [s, v.verdict])),
             reasons: [...new Set(Object.values(h.states).flatMap(v => v.reasons))] };
  });
  await click("[data-act=car]");
  return r;
}
const results = {};
for (const fault of ["", "squeal", "misfire", "weak"]) {
  results[fault || "healthy"] = [];
  for (let i = 0; i < runs; i++) results[fault || "healthy"].push(await check(fault));
}
fs.mkdirSync(path.join(here, "..", "tests", "out"), { recursive: true });
await page.screenshot({ path: path.join(here, "out", "car.png"), fullPage: true });
await browser.close();
server.close();

let bad = 0;
for (const [k, rs] of Object.entries(results)) {
  console.log(k.padEnd(8), rs.map(r => r.verdict).join(" "), "|", rs[0].reasons.join("; "), "| likely:", rs[0].likely);
  const flagged = rs.filter(r => r.verdict === "bad" || r.verdict === "warn").length;
  if (k === "healthy" && rs.filter(r => r.verdict === "bad").length > runs / 2) bad++;
  if (k !== "healthy" && flagged < runs / 2) bad++;
}
const noLikely = Object.values(results).flat().filter(r => !r.likely).length;
if (noLikely) { console.log("checks without a classifier result:", noLikely); bad++; }
if (errors.length) { console.log("page errors:", errors); bad++; }
process.exit(bad ? 1 : 0);
