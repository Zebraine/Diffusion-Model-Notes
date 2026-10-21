(function () {
  "use strict";

  var grid = document.getElementById("paper-grid");
  if (!grid) return;

  var cards = Array.prototype.slice.call(grid.querySelectorAll(".paper-card"));
  var search = document.getElementById("paper-search");
  var chips = Array.prototype.slice.call(document.querySelectorAll(".tag-chip"));
  var count = document.getElementById("result-count");
  var empty = document.getElementById("empty-state");
  var clear = document.getElementById("clear-filters");
  var activeTag = "all";

  function normalize(value) {
    return (value || "").trim().toLowerCase();
  }

  function applyFilters() {
    var query = normalize(search ? search.value : "");
    var visible = 0;

    cards.forEach(function (card) {
      var tags = (card.getAttribute("data-tags") || "").split("|");
      var haystack = card.getAttribute("data-search") || "";
      var tagMatch = activeTag === "all" || tags.indexOf(activeTag) !== -1;
      var queryMatch = !query || haystack.indexOf(query) !== -1;
      var show = tagMatch && queryMatch;

      card.hidden = !show;
      if (show) visible += 1;
    });

    if (count) {
      var filtered = activeTag !== "all" || query;
      count.textContent = filtered
        ? visible + " / " + cards.length + " 篇论文"
        : cards.length + " 篇论文";
    }

    if (empty) empty.hidden = visible !== 0;
  }

  chips.forEach(function (chip) {
    chip.setAttribute("aria-pressed", chip.classList.contains("is-active") ? "true" : "false");
    chip.addEventListener("click", function () {
      activeTag = chip.getAttribute("data-tag") || "all";
      chips.forEach(function (other) {
        var active = other === chip;
        other.classList.toggle("is-active", active);
        other.setAttribute("aria-pressed", active ? "true" : "false");
      });
      applyFilters();
    });
  });

  if (search) {
    search.addEventListener("input", applyFilters);
    search.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        search.value = "";
        applyFilters();
      }
    });
  }

  if (clear) {
    clear.addEventListener("click", function () {
      if (search) search.value = "";
      activeTag = "all";
      chips.forEach(function (chip) {
        var active = chip.getAttribute("data-tag") === "all";
        chip.classList.toggle("is-active", active);
        chip.setAttribute("aria-pressed", active ? "true" : "false");
      });
      applyFilters();
      if (search) search.focus();
    });
  }

  applyFilters();
})();
