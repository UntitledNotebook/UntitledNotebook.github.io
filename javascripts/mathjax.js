window.MathJax = {
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex",
  },
  output: {
    displayOverflow: "linebreak",
    linebreaks: {
      inline: false,
    },
  },
  startup: {
    async pageReady() {
      const mathJax = window.MathJax;
      await mathJax.startup.defaultPageReady();

      const article = document.querySelector(".md-content__inner:has(div.arithmatex)");
      if (!article) return;

      let articleWidth = article.clientWidth;
      let resizeTimer;
      const { STATE } = mathJax._.core.MathItem;

      new ResizeObserver(() => {
        const width = article.clientWidth;
        if (width === articleWidth) return;

        articleWidth = width;
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
          mathJax.startup.document.rerenderPromise(STATE.METRICS).catch((error) => {
            console.error("Failed to reflow display equations", error);
          });
        }, 150);
      }).observe(article);
    },
  },
};
