#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import MarkdownIt from "markdown-it";
import { highlightBatch, escapeHtml } from "./lib/highlight.mjs";
import { renderMarkdown } from "./lib/markdown.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const papersDir = path.join(root, "papers");
const templatesDir = path.join(here, "templates");
const checkOnly = process.argv.includes("--check");
const warnings = [];
const errors = [];

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function writeFile(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  console.log(`  wrote ${path.relative(root, file)} (${(Buffer.byteLength(content) / 1024).toFixed(1)} kB)`);
}

function number3(value) {
  return String(value).padStart(3, "0");
}

function initials(name) {
  const parts = String(name).trim().split(/\s+/).filter(Boolean);
  if (parts.length === 1) return Array.from(parts[0]).slice(0, 2).join("").toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/'/g, "&#39;");
}

function imageSize(file) {
  try {
    const buffer = fs.readFileSync(file);
    if (buffer.toString("ascii", 0, 4) === "RIFF" && buffer.toString("ascii", 8, 12) === "WEBP") {
      let offset = 12;
      while (offset + 8 <= buffer.length) {
        const fourcc = buffer.toString("ascii", offset, offset + 4);
        const size = buffer.readUInt32LE(offset + 4);
        const data = offset + 8;
        if (fourcc === "VP8X") {
          return { width: 1 + buffer.readUIntLE(data + 4, 3), height: 1 + buffer.readUIntLE(data + 7, 3) };
        }
        if (fourcc === "VP8L") {
          const bits = buffer.readUInt32LE(data + 1);
          return { width: (bits & 0x3fff) + 1, height: ((bits >> 14) & 0x3fff) + 1 };
        }
        if (fourcc === "VP8 ") {
          if (buffer[data + 3] === 0x9d && buffer[data + 4] === 0x01 && buffer[data + 5] === 0x2a) {
            return {
              width: buffer.readUInt16LE(data + 6) & 0x3fff,
              height: buffer.readUInt16LE(data + 8) & 0x3fff,
            };
          }
        }
        offset = data + size + (size % 2);
      }
    }
    if (buffer[0] === 0x89 && buffer.toString("ascii", 1, 4) === "PNG") {
      return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
    }
    if (buffer[0] === 0xff && buffer[1] === 0xd8) {
      let offset = 2;
      while (offset < buffer.length) {
        if (buffer[offset] !== 0xff) {
          offset += 1;
          continue;
        }
        const marker = buffer[offset + 1];
        if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
          return { width: buffer.readUInt16BE(offset + 7), height: buffer.readUInt16BE(offset + 5) };
        }
        offset += 2 + buffer.readUInt16BE(offset + 2);
      }
    }
    if (buffer.toString("ascii", 0, 3) === "GIF") {
      return { width: buffer.readUInt16LE(6), height: buffer.readUInt16LE(8) };
    }
    if (file.endsWith(".svg")) {
      const text = buffer.toString("utf8");
      const width = text.match(/\bwidth="([\d.]+)(?:px)?"/);
      const height = text.match(/\bheight="([\d.]+)(?:px)?"/);
      const viewBox = text.match(/\bviewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"/);
      if (width && height) return { width: Number(width[1]), height: Number(height[1]) };
      if (viewBox) return { width: Number(viewBox[1]), height: Number(viewBox[2]) };
    }
  } catch (error) {
    warnings.push(`Could not read image size for ${path.relative(root, file)}: ${error.message}`);
  }
  return null;
}

function loadPapers() {
  return fs
    .readdirSync(papersDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith(".") && entry.name !== "assets")
    .map((entry) => {
      const file = path.join(papersDir, entry.name, "paper.json");
      if (!fs.existsSync(file)) return null;
      const data = readJson(file);
      return { ...data, slug: data.slug || entry.name, dir: path.join(papersDir, entry.name), metaFile: file };
    })
    .filter(Boolean)
    .sort((a, b) => (a.order ?? 999) - (b.order ?? 999));
}

function normalizePaper(paper) {
  const number = number3(paper.order ?? 0);
  const coverFile = paper.cover ? path.join(paper.dir, paper.cover) : null;
  const dims = coverFile && fs.existsSync(coverFile) ? imageSize(coverFile) : null;
  const stats = {
    sections: 0,
    figures: 0,
    tables: 0,
    formulas: 0,
    code_blocks: 0,
    algorithms: 0,
    reading_minutes: paper.reading_minutes ?? 20,
    ...(paper.stats || {}),
  };
  return {
    ...paper,
    number,
    tags: paper.tags || [],
    authors: paper.authors || [],
    stats,
    coverFile,
    coverDims: dims,
    pageFile: path.join(paper.dir, "index.html"),
    url: `${paper.slug}/index.html`,
    coverUrl: paper.cover || "",
  };
}

function renderAuthors(paper) {
  return paper.authors
    .map((author) => {
      const name = typeof author === "string" ? author : author.name;
      const url = typeof author === "string" ? "" : author.url;
      const label = escapeHtml(name);
      return url
        ? `<a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
        : `<span>${label}</span>`;
    })
    .join('<span class="author-sep" aria-hidden="true">、</span>');
}

function renderAuthorInitials(paper) {
  return paper.authors
    .map((author) => `<span class="author-initial">${escapeHtml(initials(typeof author === "string" ? author : author.name))}</span>`)
    .join("");
}

function heroButtons(paper) {
  const buttons = [];
  if (paper.pdf_url) {
    buttons.push(`<a class="button button-ghost" href="${escapeAttr(paper.pdf_url)}" target="_blank" rel="noopener noreferrer">PDF 原文 ↗</a>`);
  }
  const sourceButtonLabel = paper.source_button_label || "英文摘要";
  buttons.push(`<a class="button button-ghost" href="${escapeAttr(paper.source_url || "#")}" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceButtonLabel)} ↗</a>`);
  return buttons.join("\n            ");
}

function heroStats(paper) {
  const stats = paper.stats;
  const items = [
    [stats.sections, "章节 · Sections"],
    [stats.figures, "图示 · Figures"],
    [stats.equations ?? stats.formulas, "公式 · Equations"],
    stats.algorithms
      ? [stats.algorithms, "算法 · Algorithms"]
      : [stats.code_blocks, "代码 · Snippets"],
  ];
  return items
    .map(([value, label]) => `<div class="hero-stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("\n            ");
}

function sourceActions(paper) {
  const actions = [];
  if (paper.source_url) {
    const sourceActionLabel = paper.source_action_label || "阅读英文原文";
    actions.push(`<a class="button button-primary" href="${escapeAttr(paper.source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceActionLabel)} ↗</a>`);
  }
  if (paper.pdf_url) {
    actions.push(`<a class="button button-ghost" href="${escapeAttr(paper.pdf_url)}" target="_blank" rel="noopener noreferrer">下载 PDF ↗</a>`);
  }
  if (paper.code_url) {
    actions.push(`<a class="button button-ghost" href="${escapeAttr(paper.code_url)}" target="_blank" rel="noopener noreferrer">官方实现 ↗</a>`);
  }
  return actions.join("\n              ");
}

function seriesNav(all, index) {
  const current = all[index];
  const prev = all[index - 1];
  const next = all[index + 1];
  const links = [];

  if (prev) {
    links.push(`<a class="series-link" href="../${escapeAttr(prev.url)}">
    <span class="series-label">上一篇 · ${escapeHtml(prev.number)}</span>
    <span class="series-title">${escapeHtml(prev.title)}</span>
  </a>`);
  } else {
    links.push(`<a class="series-link" href="../index.html">
    <span class="series-label">论文库</span>
    <span class="series-title">全部论文</span>
  </a>`);
  }

  if (next) {
    links.push(`<a class="series-link series-link--next" href="../${escapeAttr(next.url)}">
    <span class="series-label">下一篇 · ${escapeHtml(next.number)}</span>
    <span class="series-title">${escapeHtml(next.title)}</span>
    <span class="series-arrow" aria-hidden="true">→</span>
  </a>`);
  } else {
    links.push(`<a class="series-link series-link--next" href="../index.html">
    <span class="series-label">论文库</span>
    <span class="series-title">返回全部论文</span>
    <span class="series-arrow" aria-hidden="true">→</span>
  </a>`);
  }

  return `<nav class="series-nav" aria-label="论文系列导航">
  ${links.join("\n  ")}
</nav>`;
}

function parseFrontTitle(paper) {
  const title = paper.title || "未命名论文";
  const lead = paper.title_lead ?? title;
  const accent = paper.title_accent ?? "";
  return { title, lead, accent };
}

function renderPaper(paper, all, rendered) {
  const template = fs.readFileSync(path.join(templatesDir, "paper.html"), "utf8");
  const { title, lead, accent } = parseFrontTitle(paper);
  const index = all.findIndex((item) => item.slug === paper.slug);
  const cover = paper.cover || "";
  const coverDims = paper.coverDims || { width: 1200, height: 400 };
  const reading = paper.reading_minutes || paper.stats.reading_minutes || 20;
  const sourceMeta = [
    paper.authors.map((a) => (typeof a === "string" ? a : a.name)).join("、"),
    paper.venue,
    paper.date ? `${paper.date.slice(0, 4)} 年 ${Number(paper.date.slice(5, 7))} 月` : "",
  ]
    .filter(Boolean)
    .join(" · ");

  const values = {
    TITLE: escapeHtml(title),
    TITLE_EN: escapeHtml(paper.title_en || ""),
    DESCRIPTION: escapeAttr(paper.dek || paper.summary || ""),
    AUTHOR_META: escapeAttr(paper.authors.map((a) => (typeof a === "string" ? a : a.name)).join(", ")),
    NUMBER: escapeHtml(paper.number),
    SERIES: escapeHtml(paper.series || paper.venue || "Paper"),
    KIND_CN: escapeHtml(paper.kind || "中文精译"),
    TITLE_LEAD: escapeHtml(lead),
    TITLE_ACCENT: escapeHtml(accent),
    DEK: escapeHtml(paper.dek || paper.summary || ""),
    AUTHOR_INITIALS: renderAuthorInitials(paper),
    AUTHOR_LINKS: renderAuthors(paper),
    DATE: escapeAttr(paper.date || ""),
    DATE_DISPLAY: escapeHtml((paper.date || "").replaceAll("-", ".")),
    VENUE: escapeHtml(paper.venue || ""),
    READING: escapeHtml(reading),
    HERO_BUTTONS: heroButtons(paper),
    HERO_STATS: heroStats(paper),
    COVER: escapeAttr(cover),
    COVER_ALT: escapeAttr(paper.cover_alt || title),
    COVER_W: String(coverDims.width),
    COVER_H: String(coverDims.height),
    COVER_RATIO: escapeAttr(paper.cover_ratio || "3x1"),
    COVER_CAPTION: escapeHtml(paper.cover_caption || ""),
    COVER_META: escapeHtml(paper.cover_meta || `${paper.venue || ""} · ${paper.year || ""}`),
    NOTE_LABEL: escapeHtml(paper.note_label || "翻译说明"),
    NOTE_SEAL: escapeHtml(paper.note_seal || "译"),
    TRANSLATION_NOTE: paper.translation_note || "公式编号沿用原文；为便于中文阅读，正文引用改为作者-年份。图表、表格与算法已重新排版。正文为全译，附录为导读与关键结论选译。",
    SOURCE_LABEL: escapeHtml(paper.source_label || "英文原文"),
    CONTENT: rendered.html,
    TOC: rendered.toc.aside,
    MOBILE_TOC: rendered.toc.mobile,
    SERIES_NAV: seriesNav(all, index),
    SOURCE_META: escapeHtml(sourceMeta),
    SOURCE_ACTIONS: sourceActions(paper),
    SOURCE_URL: escapeAttr(paper.source_url || "#"),
    DISCLAIMER: escapeHtml(paper.disclaimer || "本页面为学习交流用途的中文翻译。原文、图表及相关材料的版权归原作者所有。"),
    FOOTER_NOTE: escapeHtml(`${paper.kind || "中文精译"} · ${paper.venue || ""}`),
  };

  let html = template;
  for (const [key, value] of Object.entries(values)) {
    html = html.replaceAll(`{{${key}}}`, value);
  }
  const leftovers = html.match(/\{\{[A-Z_]+\}\}/g);
  if (leftovers) errors.push(`${paper.slug}: unreplaced template keys ${[...new Set(leftovers)].join(", ")}`);
  return html;
}

function tagChips(papers) {
  const counts = new Map();
  for (const paper of papers) {
    for (const tag of paper.tags) counts.set(tag, (counts.get(tag) || 0) + 1);
  }
  const tags = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "zh-CN"));
  return [
    '<button class="tag-chip is-active" type="button" data-tag="all">全部</button>',
    ...tags.map(
      ([tag, count]) =>
        `<button class="tag-chip" type="button" data-tag="${escapeAttr(tag)}">${escapeHtml(tag)} <span aria-hidden="true">${count}</span></button>`,
    ),
  ].join("\n          ");
}

function paperCard(paper, index) {
  const stats = paper.stats;
  const search = [
    paper.title,
    paper.title_en,
    paper.summary,
    paper.dek,
    paper.venue,
    paper.year,
    ...paper.authors.map((a) => (typeof a === "string" ? a : a.name)),
    ...paper.tags,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  const tags = paper.tags.map((tag) => `<li><span>${escapeHtml(tag)}</span></li>`).join("");
  return `<article class="paper-card" data-tags="${escapeAttr(paper.tags.join("|"))}" data-search="${escapeAttr(search)}">
  <a class="paper-card-cover" data-fit="${escapeAttr(paper.cover_fit || "cover")}" href="${escapeAttr(paper.url)}" aria-label="阅读《${escapeAttr(paper.title)}》">
    <img src="${escapeAttr(`${paper.slug}/${paper.cover}`)}" alt="${escapeAttr(paper.cover_alt || paper.title)}" width="${paper.coverDims?.width || ""}" height="${paper.coverDims?.height || ""}" loading="${index < 3 ? "eager" : "lazy"}" decoding="${index < 3 ? "sync" : "async"}">
    <span class="paper-card-index">${escapeHtml(paper.number)}</span>
    <span class="paper-card-kind">${escapeHtml(paper.kind || "精译")}</span>
  </a>
  <div class="paper-card-body">
    <div class="paper-card-topline">
      <span>${escapeHtml(paper.venue || "")}</span>
      <span>${escapeHtml(paper.year || "")}</span>
    </div>
    <h3><a href="${escapeAttr(paper.url)}">${escapeHtml(paper.title)}</a></h3>
    <p class="paper-card-en">${escapeHtml(paper.title_en || "")}</p>
    <p class="paper-card-summary">${escapeHtml(paper.summary || paper.dek || "")}</p>
    <ul class="paper-card-tags">${tags}</ul>
    <div class="paper-card-foot">
      <span>${stats.sections} 章 · ${stats.figures} 图 · ${stats.equations ?? stats.formulas} 式</span>
      <span class="paper-card-go">阅读</span>
    </div>
  </div>
</article>`;
}

function featureCard(paper) {
  if (!paper) return "";
  return `<a class="feature-card" data-fit="${escapeAttr(paper.cover_fit || "cover")}" href="${escapeAttr(paper.url)}">
  <span class="feature-cover">
    <img src="${escapeAttr(`${paper.slug}/${paper.cover}`)}" alt="${escapeAttr(paper.cover_alt || paper.title)}" width="${paper.coverDims?.width || ""}" height="${paper.coverDims?.height || ""}" loading="eager" decoding="sync">
    <span class="feature-index">${escapeHtml(paper.number)}</span>
  </span>
  <span class="feature-body">
    <span class="feature-label">最新收录 / ${escapeHtml(paper.kind || "精译")}</span>
    <span class="feature-title">${escapeHtml(paper.title)}</span>
    <span class="feature-en">${escapeHtml(paper.title_en || "")}</span>
    <span class="feature-foot"><span>${escapeHtml(paper.venue || "")} · ${escapeHtml(paper.year || "")}</span><span class="feature-arrow" aria-hidden="true">→</span></span>
  </span>
</a>`;
}

function homeStats(papers) {
  const total = (key) => papers.reduce((sum, paper) => sum + (Number(paper.stats[key]) || 0), 0);
  const items = [
    [papers.length, "论文 · Papers"],
    [total("sections"), "章节 · Sections"],
    [total("figures"), "图示 · Figures"],
    [papers.reduce((sum, paper) => sum + (Number(paper.stats.equations ?? paper.stats.formulas) || 0), 0), "公式 · Equations"],
  ];
  return items
    .map(([value, label]) => `<div class="home-stat"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("\n            ");
}

function renderHome(papers) {
  const template = fs.readFileSync(path.join(templatesDir, "home.html"), "utf8");
  const latest = papers[papers.length - 1];
  const values = {
    HOME_STATS: homeStats(papers),
    FEATURE_CARD: featureCard(latest),
    PAPER_COUNT: `${papers.length} 篇论文`,
    TAG_CHIPS: tagChips(papers),
    LIBRARY_CARDS: papers.map((paper, index) => paperCard(paper, index)).join("\n\n"),
  };
  let html = template;
  for (const [key, value] of Object.entries(values)) {
    html = html.replaceAll(`{{${key}}}`, value);
  }
  const leftovers = html.match(/\{\{[A-Z_]+\}\}/g);
  if (leftovers) errors.push(`homepage: unreplaced template keys ${[...new Set(leftovers)].join(", ")}`);
  return html;
}

function scanFences(source) {
  const scanner = new MarkdownIt({ html: true });
  const tokens = scanner.parse(source, {});
  return tokens.filter((token) => token.type === "fence");
}

function buildPaperPage(paper, all) {
  const contentFile = path.join(paper.dir, "content.md");
  if (!fs.existsSync(contentFile)) {
    errors.push(`${paper.slug}: build requested but ${path.relative(root, contentFile)} is missing`);
    return;
  }
  console.log(`\nBuilding ${paper.slug}`);
  const source = fs.readFileSync(contentFile, "utf8");
  const fences = scanFences(source);
  const highlighted = highlightBatch(
    fences.map((token) => ({
      code: token.content,
      lang: (token.info || "").trim().split(/\s+/)[0].toLowerCase(),
    })),
  );
  const map = new Map();
  fences.forEach((token, index) => {
    const lang = (token.info || "").trim().split(/\s+/)[0].toLowerCase();
    map.set(`${lang}\u0000${token.content}`, highlighted[index]);
  });
  const rendered = renderMarkdown(source, { highlighted: map, fenceCount: fences.length });
  paper.stats = { ...paper.stats, ...rendered.stats };
  if (paper.reading_minutes) paper.stats.reading_minutes = paper.reading_minutes;
  const html = renderPaper(paper, all, rendered);
  writeFile(paper.pageFile, html);
}

function validateLocalRefs(file) {
  if (!fs.existsSync(file)) return;
  const html = fs.readFileSync(file, "utf8");
  const baseDir = path.dirname(file);
  const refs = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((match) => match[1]);
  for (const ref of refs) {
    if (!ref || ref.startsWith("#") || ref.startsWith("http://") || ref.startsWith("https://")) continue;
    if (ref.startsWith("mailto:") || ref.startsWith("data:") || ref.startsWith("javascript:")) continue;
    const clean = ref.split("#")[0].split("?")[0];
    if (!clean) continue;
    let target;
    try {
      target = decodeURIComponent(clean);
    } catch {
      target = clean;
    }
    const resolved = path.resolve(baseDir, target);
    if (!fs.existsSync(resolved)) errors.push(`${path.relative(root, file)}: missing local reference ${ref}`);
  }
}

function main() {
  console.log("Diffusion Model Notes · paper library build");
  const papers = loadPapers().map(normalizePaper);

  if (!papers.length) {
    console.error("No papers found under papers/*/paper.json");
    process.exit(1);
  }

  if (checkOnly) {
    for (const paper of papers) {
      if (!fs.existsSync(paper.pageFile)) errors.push(`${paper.slug}: missing ${path.relative(root, paper.pageFile)}`);
      if (paper.coverFile && !fs.existsSync(paper.coverFile)) errors.push(`${paper.slug}: missing cover ${path.relative(root, paper.coverFile)}`);
      validateLocalRefs(paper.pageFile);
    }
    validateLocalRefs(path.join(papersDir, "index.html"));
    return finish();
  }

  for (const paper of papers) {
    if (paper.build) buildPaperPage(paper, papers);
  }

  console.log("\nBuilding homepage");
  writeFile(path.join(papersDir, "index.html"), renderHome(papers));

  console.log("\nValidating local references");
  validateLocalRefs(path.join(papersDir, "index.html"));
  for (const paper of papers) validateLocalRefs(paper.pageFile);

  finish();
}

function finish() {
  for (const warning of warnings) console.warn(`warning: ${warning}`);
  for (const error of errors) console.error(`error: ${error}`);
  if (errors.length) {
    console.error(`\nBuild finished with ${errors.length} error(s).`);
    process.exit(1);
  }
  console.log("\nDone.");
}

main();
