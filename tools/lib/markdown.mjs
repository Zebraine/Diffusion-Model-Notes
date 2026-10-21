import katex from "katex";
import MarkdownIt from "markdown-it";
import { escapeHtml } from "./highlight.mjs";

const INLINE_OPEN = "\uE000";
const INLINE_CLOSE = "\uE001";

const MACROS = {
  "\\R": "\\mathbb{R}",
  "\\E": "\\mathbb{E}",
  "\\N": "\\mathcal{N}",
  "\\KL": "D_{\\mathrm{KL}}",
};

/**
 * Render one TeX expression with KaTeX. We keep both the MathML and the HTML
 * output so the result matches the hand-built pages and stays accessible.
 */
export function renderMath(tex, display = false) {
  return katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
    strict: "ignore",
    trust: false,
    output: "htmlAndMathml",
    macros: MACROS,
  });
}

function isEscaped(text, index) {
  let count = 0;
  for (let i = index - 1; i >= 0 && text[i] === "\\"; i -= 1) count += 1;
  return count % 2 === 1;
}

/**
 * Replace `$...$` and `$$...$$` with placeholders before Markdown runs.
 * Placeholders survive raw HTML blocks too, so figures, tables and algorithm
 * boxes can contain formulas. Fenced code and inline code are left untouched.
 */
function splitDisplayTag(tex) {
  const match = tex.match(/\\tag\*?\{([^{}]*)\}\s*$/);
  if (!match) return { body: tex, tag: "" };
  return { body: tex.slice(0, match.index).trim(), tag: match[1].trim() };
}

export function preprocessMath(source) {
  const items = [];
  const lines = source.split("\n");
  const output = [];
  let inFence = false;
  let fenceChar = "";

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const fence = line.match(/^\s*(`{3,}|~{3,})/);

    if (fence) {
      if (!inFence) {
        inFence = true;
        fenceChar = fence[1][0];
        output.push(line);
        continue;
      }
      if (fence[1][0] === fenceChar) {
        inFence = false;
        output.push(line);
        continue;
      }
    }

    if (inFence) {
      output.push(line);
      continue;
    }

    const trimmed = line.trim();
    if (trimmed.startsWith("$$")) {
      const rest = trimmed.slice(2);
      if (rest.endsWith("$$") && rest.length >= 2) {
        const tex = rest.slice(0, -2).trim();
        if (tex) {
          const parsed = splitDisplayTag(tex);
          const id = items.push({ tex: parsed.body, display: true, tag: parsed.tag, html: renderMath(parsed.body, true) }) - 1;
          output.push("", `<div class="math-slot" data-math="${id}"></div>`, "");
          continue;
        }
      } else {
        const buffer = [rest];
        let closed = false;
        let j = i + 1;
        for (; j < lines.length; j += 1) {
          const current = lines[j];
          const end = current.indexOf("$$");
          if (end !== -1) {
            buffer.push(current.slice(0, end));
            const after = current.slice(end + 2).trim();
            if (after) buffer.push(after);
            closed = true;
            break;
          }
          buffer.push(current);
        }
        if (closed) {
          const tex = buffer.join("\n").trim();
          if (tex) {
            const parsed = splitDisplayTag(tex);
            const id = items.push({ tex: parsed.body, display: true, tag: parsed.tag, html: renderMath(parsed.body, true) }) - 1;
            output.push("", `<div class="math-slot" data-math="${id}"></div>`, "");
            i = j;
            continue;
          }
        }
      }
    }

    output.push(replaceInlineMath(line, items));
  }

  return { source: output.join("\n"), items };
}

function replaceInlineMath(line, items) {
  let result = "";
  let i = 0;
  let inTag = false;

  while (i < line.length) {
    const char = line[i];

    // Leave HTML tag attributes alone: a KaTeX fragment inside e.g.
    // data-caption would otherwise break the attribute quoting.
    if (!inTag && char === "<" && /[A-Za-z/!]/.test(line[i + 1] || "")) {
      inTag = true;
      result += char;
      i += 1;
      continue;
    }
    if (inTag) {
      result += char;
      if (char === ">") inTag = false;
      i += 1;
      continue;
    }

    if (char === "`") {
      let run = 0;
      while (line[i + run] === "`") run += 1;
      const marker = "`".repeat(run);
      const end = line.indexOf(marker, i + run);
      if (end === -1) {
        result += line.slice(i);
        break;
      }
      result += line.slice(i, end + run);
      i = end + run;
      continue;
    }

    if (char === "\\" && line[i + 1] === "$") {
      result += "\\$";
      i += 2;
      continue;
    }

    if (char === "$" && !isEscaped(line, i)) {
      let j = i + 1;
      while (j < line.length) {
        if (line[j] === "$" && !isEscaped(line, j)) break;
        j += 1;
      }
      if (j < line.length) {
        const tex = line.slice(i + 1, j).trim();
        if (tex) {
          const id = items.push({ tex, display: false, html: renderMath(tex, false) }) - 1;
          result += `${INLINE_OPEN}${id}${INLINE_CLOSE}`;
          i = j + 1;
          continue;
        }
      }
    }

    result += char;
    i += 1;
  }

  return result;
}

function restoreMath(html, items) {
  let result = html.replace(
    /<div class="math-slot" data-math="(\d+)">\s*<\/div>/g,
    (match, id) => {
      const item = items[Number(id)];
      if (!item) return match;
      const tag = item.tag ? `<span class="math-tag">(${escapeHtml(item.tag)})</span>` : "";
      return `<div class="math-display math-display--split" role="math"><div class="math-scroll">${item.html}</div>${tag}</div>`;
    },
  );

  result = result.replace(
    new RegExp(`${INLINE_OPEN}(\\d+)${INLINE_CLOSE}`, "g"),
    (match, id) => {
      const item = items[Number(id)];
      if (!item) return match;
      return `<span class="math">${item.html}</span>`;
    },
  );

  return result;
}

export function slugify(text) {
  const slug = String(text)
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/&[a-z]+;/g, " ")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return slug || "section";
}

export function stripInline(text) {
  return String(text)
    .replace(new RegExp(`${INLINE_OPEN}\\d+${INLINE_CLOSE}`, "g"), "")
    .replace(/<[^>]+>/g, "")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[`*~]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function prettyLang(lang) {
  const names = {
    python: "Python",
    py: "Python",
    bash: "Bash",
    sh: "Bash",
    shell: "Bash",
    console: "Bash",
    js: "JavaScript",
    javascript: "JavaScript",
    ts: "TypeScript",
    typescript: "TypeScript",
    json: "JSON",
    yaml: "YAML",
    yml: "YAML",
    text: "Text",
    plaintext: "Text",
    latex: "LaTeX",
    tex: "LaTeX",
    html: "HTML",
    css: "CSS",
    sql: "SQL",
    diff: "Diff",
  };
  return names[(lang || "").toLowerCase()] || (lang ? lang.replace(/^\w/, (c) => c.toUpperCase()) : "Text");
}

function fenceHtml(info, code, highlighted) {
  const lang = (info || "").trim().split(/\s+/)[0].toLowerCase() || "text";
  const label = prettyLang(lang);
  const body = highlighted ?? escapeHtml(code);
  return `<div class="code-block" role="group" aria-label="${escapeHtml(label)} 代码示例">
<div class="code-head"><span class="code-lang">${escapeHtml(label)}</span><span class="code-actions"></span></div>
<pre class="code-body"><code class="language-${escapeHtml(lang)}">${body}</code></pre>
</div>`;
}

function tocLink(heading) {
  const title = escapeHtml(heading.text);
  if (heading.level === 3) {
    return `<a class="toc-sub" href="#${escapeHtml(heading.id)}"><span>${title}</span></a>`;
  }
  return `<a href="#${escapeHtml(heading.id)}"><span class="toc-num">${escapeHtml(heading.num)}</span><span>${title}</span></a>`;
}

export function renderToc(headings) {
  const links = headings.map(tocLink).join("\n");
  const aside = `<aside class="toc">
  <div class="toc-inner">
    <p class="toc-label">目录 <span>CONTENTS</span></p>
    <nav>
${links}
    </nav>
    <div class="toc-foot"><span>阅读进度</span><span class="toc-percent" data-toc-percent>0%</span></div>
  </div>
</aside>`;

  const mobile = `<details class="mobile-toc">
  <summary><span>本文目录</span><span class="summary-arrow" aria-hidden="true">⌄</span></summary>
  <nav>
${links}
  </nav>
</details>`;

  return { aside, mobile };
}

function countMatches(html, pattern) {
  return (html.match(pattern) || []).length;
}

export function contentStats(html, headings, mathItems, fenceCount, source) {
  const text = html
    .replace(/<pre[\s\S]*?<\/pre>/g, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z]+;/g, " ");
  const cjk = (text.match(/[\u3400-\u9fff]/g) || []).length;
  const words = (text.match(/[A-Za-z][A-Za-z0-9'-]*/g) || []).length;
  const minutes = Math.max(5, Math.ceil(cjk / 380 + words / 220 + fenceCount * 0.2));

  return {
    sections: headings.filter((h) => h.level === 2).length,
    figures: countMatches(html, /<figure\b/g),
    tables: countMatches(html, /<table\b/g),
    formulas: mathItems.length,
    equations: mathItems.filter((item) => item.display).length,
    code_blocks: fenceCount,
    algorithms: countMatches(html, /class="algorithm"/g),
    reading_minutes: minutes,
    characters: cjk,
    source_length: source.length,
  };
}

/**
 * Convert one paper's Markdown into the shared article HTML.
 * @param {string} source
 * @param {{highlighted:Map<string,string>}} env
 */
export function renderMarkdown(source, env) {
  const math = preprocessMath(source);
  const md = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: false,
    breaks: false,
  });

  // Pull `{#custom-id}` off headings before inline parsing.
  md.core.ruler.before("inline", "heading_ids", (state) => {
    const tokens = state.tokens;
    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i].type !== "heading_open") continue;
      const inline = tokens[i + 1];
      if (!inline || inline.type !== "inline") continue;
      const match = inline.content.match(/\s*\{#([A-Za-z0-9_-]+)\}\s*$/);
      if (match) {
        inline.content = inline.content.slice(0, match.index).trimEnd();
        tokens[i].meta = { ...(tokens[i].meta || {}), id: match[1] };
      }
    }
  });

  // Assign ids, section numbers and collect the table of contents.
  md.core.ruler.push("heading_toc", (state) => {
    const headings = [];
    let h2 = 0;
    const tokens = state.tokens;

    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i].type !== "heading_open") continue;
      const open = tokens[i];
      const inline = tokens[i + 1];
      const close = tokens[i + 2];
      const level = Number(open.tag.slice(1));
      const text = stripInline(inline.content) || "未命名章节";
      const id = (open.meta && open.meta.id) || slugify(text) || `section-${i}`;
      const num = level === 2 ? String((h2 += 1)).padStart(2, "0") : "";
      const meta = { id, text, level, num };

      open.attrSet("id", id);
      open.meta = { ...(open.meta || {}), ...meta };
      if (close && close.type === "heading_close") close.meta = { ...(close.meta || {}), ...meta };

      headings.push(meta);
    }

    state.env.headings = headings;
  });

  md.renderer.rules.heading_open = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    if (token.tag === "h2") {
      const meta = token.meta || {};
      return `<h2 id="${escapeHtml(meta.id)}"><span class="h2-num" aria-hidden="true">${escapeHtml(meta.num)}</span>`;
    }
    if (token.tag === "h3") {
      return `<h3 id="${escapeHtml((token.meta || {}).id)}">`;
    }
    return self.renderToken(tokens, idx, options);
  };

  md.renderer.rules.heading_close = (tokens, idx) => {
    const token = tokens[idx];
    const meta = token.meta || {};
    if (token.tag === "h2" || token.tag === "h3") {
      return ` <a class="heading-anchor" href="#${escapeHtml(meta.id)}" aria-label="本节永久链接" title="本节链接">#</a></${token.tag}>`;
    }
    return `</${token.tag}>`;
  };

  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    const href = tokens[idx].attrGet("href") || "";
    if (/^https?:\/\//i.test(href)) {
      tokens[idx].attrSet("target", "_blank");
      tokens[idx].attrSet("rel", "noopener noreferrer");
    }
    return self.renderToken(tokens, idx, options);
  };

  const defaultImage = md.renderer.rules.image;
  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    tokens[idx].attrSet("loading", "lazy");
    tokens[idx].attrSet("decoding", "async");
    return defaultImage(tokens, idx, options, env, self);
  };

  md.renderer.rules.table_open = () => '<div class="table-wrap"><table class="paper-table">';
  md.renderer.rules.table_close = () => "</table></div>";

  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const lang = (token.info || "").trim().split(/\s+/)[0].toLowerCase();
    const key = `${lang}\u0000${token.content}`;
    return fenceHtml(token.info, token.content, env.highlighted.get(key));
  };

  const renderEnv = { highlighted: env.highlighted, fenceCount: env.fenceCount };
  const htmlRaw = md.render(math.source, renderEnv);
  const headings = renderEnv.headings || [];
  const html = restoreMath(htmlRaw, math.items);
  const stats = contentStats(html, headings, math.items, env.fenceCount, source);

  return {
    html,
    headings,
    stats,
    toc: renderToc(headings),
    mathCount: math.items.length,
  };
}
