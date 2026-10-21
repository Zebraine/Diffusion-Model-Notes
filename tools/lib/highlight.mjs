import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const helper = path.join(here, "highlight.py");

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Highlight every fenced block in one Python process.
 * @param {{code:string, lang:string}[]} items
 * @returns {string[]}
 */
export function highlightBatch(items) {
  if (!items.length) return [];
  try {
    const stdout = execFileSync("python3", [helper], {
      input: JSON.stringify(items),
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    });
    const result = JSON.parse(stdout);
    if (!Array.isArray(result) || result.length !== items.length) {
      throw new Error("unexpected Pygments output");
    }
    return result;
  } catch (error) {
    console.warn(`[highlight] Pygments failed (${error.message}); using plain text.`);
    return items.map((item) => escapeHtml(item.code));
  }
}
