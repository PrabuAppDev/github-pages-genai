---
layout: default
title: RAG-tutorial
---

# Tutorials on RAG

<a href="https://github.com/PrabuAppDev/genai-rag/blob/main/rag-101.md" target="_blank">View the original RAG tutorial on GitHub</a>

---

<div id="rag-content" style="width:100%; height:600px; overflow:auto;">
  Loading RAG tutorial...
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/showdown/1.9.1/showdown.min.js"></script>
<script>
  document.addEventListener('DOMContentLoaded', function () {
      fetch('https://raw.githubusercontent.com/PrabuAppDev/genai-rag/main/rag-101.md')
          .then(response => response.text())
          .then(markdown => {
              const converter = new showdown.Converter();
              const htmlContent = converter.makeHtml(markdown);
              document.getElementById('rag-content').innerHTML = htmlContent;
          })
          .catch(error => {
              document.getElementById('rag-content').innerHTML = 'Error loading content.';
          });
  });
</script>
