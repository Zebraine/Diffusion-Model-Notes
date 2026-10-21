(function () {
  "use strict";

  var doc = document.documentElement;
  var body = document.body;
  var article = document.getElementById("article");
  if (!article) return;
  var progressBar = document.getElementById("progress-bar");
  var tocPercent = document.querySelectorAll("[data-toc-percent]");
  var backToTop = document.querySelector(".back-to-top");
  var dialog = document.getElementById("lightbox");
  var dialogImage = document.getElementById("lightbox-image");
  var dialogCaption = document.getElementById("lightbox-caption");

  doc.classList.add("js");

  /* Keep external references from hijacking the reading page. */
  article.querySelectorAll('a[href^="http"]').forEach(function (link) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });

  /* Copy buttons are injected here so they never appear as dead controls. */
  document.querySelectorAll(".code-block").forEach(function (block) {
    var head = block.querySelector(".code-head");
    var code = block.querySelector("code");
    if (!head || !code) return;

    var actions = head.querySelector(".code-actions");
    if (!actions) {
      actions = document.createElement("span");
      actions.className = "code-actions";
      head.appendChild(actions);
    }

    var button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.textContent = "复制代码";
    button.setAttribute("aria-label", "复制这段代码");

    button.addEventListener("click", function () {
      var text = code.innerText.replace(/\s+$/, "");

      function onSuccess() {
        button.textContent = "已复制";
        button.classList.add("is-copied");
        window.setTimeout(function () {
          button.textContent = "复制代码";
          button.classList.remove("is-copied");
        }, 1600);
      }

      function onFailure() {
        button.textContent = "复制失败";
        window.setTimeout(function () {
          button.textContent = "复制代码";
        }, 1600);
      }

      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(onSuccess, function () {
          fallbackCopy(text) ? onSuccess() : onFailure();
        });
      } else {
        fallbackCopy(text) ? onSuccess() : onFailure();
      }
    });

    actions.appendChild(button);
  });

  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    area.style.opacity = "0";
    body.appendChild(area);
    area.select();

    var copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }

    area.remove();
    return copied;
  }


  /* Horizontal overflow areas need keyboard access to their scrollbars. */
  var scrollRegions = document.querySelectorAll(
    ".code-body, .math-display, .math-scroll, .output-block pre"
  );

  function syncScrollableRegions() {
    scrollRegions.forEach(function (region) {
      if (region.scrollWidth > region.clientWidth + 1 || region.scrollHeight > region.clientHeight + 1) {
        region.tabIndex = 0;
      } else {
        region.removeAttribute("tabindex");
      }
    });
  }

  window.addEventListener("load", syncScrollableRegions);
  window.addEventListener("resize", syncScrollableRegions);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(syncScrollableRegions);
  }
  syncScrollableRegions();

  /* Reading progress and active table-of-contents entries. */
  var tocLinks = Array.prototype.slice.call(
    document.querySelectorAll('.toc a[href^="#"], .mobile-toc a[href^="#"]')
  );

  var sections = tocLinks
    .map(function (link) {
      var id = link.getAttribute("href");
      return id && id.length > 1 ? document.querySelector(id) : null;
    })
    .filter(function (section, index, list) {
      return section && list.indexOf(section) === index;
    });

  function updateScrollState() {
    var scrollTop = window.scrollY || window.pageYOffset;
    var articleTop = article.getBoundingClientRect().top + scrollTop;
    var articleHeight = article.offsetHeight;
    var viewport = window.innerHeight;
    var progress = 0;

    if (articleHeight > 0) {
      progress = (scrollTop - articleTop + viewport * 0.16) / articleHeight;
      progress = Math.min(1, Math.max(0, progress));
    }

    if (progressBar) {
      progressBar.style.transform = "scaleX(" + progress.toFixed(4) + ")";
    }

    tocPercent.forEach(function (node) {
      node.textContent = Math.round(progress * 100) + "%";
    });

    var threshold = viewport * 0.3;
    var currentId = "";

    sections.forEach(function (section) {
      if (section.getBoundingClientRect().top <= threshold) {
        currentId = section.id;
      }
    });

    if (!currentId && sections.length) {
      currentId = sections[0].id;
    }

    tocLinks.forEach(function (link) {
      var active = link.getAttribute("href") === "#" + currentId;
      link.classList.toggle("is-active", active);
      if (active) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    });

    if (backToTop) {
      backToTop.classList.toggle("is-visible", scrollTop > 700);
    }

  }

  var ticking = false;
  function requestScrollUpdate() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () {
      updateScrollState();
      ticking = false;
    });
  }

  window.addEventListener("scroll", requestScrollUpdate, { passive: true });
  window.addEventListener("resize", requestScrollUpdate);
  updateScrollState();

  if (backToTop) {
    backToTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  document.querySelectorAll(".mobile-toc a").forEach(function (link) {
    link.addEventListener("click", function () {
      var details = link.closest("details");
      if (details) details.open = false;
    });
  });

  /* Image lightbox: links keep working when dialog support is unavailable. */
  if (dialog && dialogImage && typeof dialog.showModal === "function") {
    document.querySelectorAll(".zoom-link").forEach(function (link) {
      link.addEventListener("click", function (event) {
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
          return;
        }

        event.preventDefault();
        var image = link.querySelector("img");
        dialogImage.src = link.getAttribute("href");
        dialogImage.alt = image ? image.alt : "";
        if (dialogCaption) {
          dialogCaption.textContent = link.getAttribute("data-caption") || "";
        }
        dialog.showModal();
        body.classList.add("lightbox-open");
      });
    });

    function closeLightbox() {
      if (dialog.open) dialog.close();
      body.classList.remove("lightbox-open");
      dialogImage.removeAttribute("src");
    }

    dialog.querySelectorAll("[data-close-lightbox]").forEach(function (button) {
      button.addEventListener("click", closeLightbox);
    });

    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) {
        closeLightbox();
      }
    });

    dialog.addEventListener("cancel", function () {
      body.classList.remove("lightbox-open");
    });

    dialog.addEventListener("close", function () {
      body.classList.remove("lightbox-open");
      dialogImage.removeAttribute("src");
    });
  }

  /* A small, quiet guard against stale scroll position on reload. */
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "auto";
  }

})();
