pandoc paper.md -o index.html \
	--css style.css \
	--mathjax \
	--filter pandoc-fignos \
	--filter pandoc-tablenos \
	--filter pandoc-citeproc \
	--bibliography references.bib \
	-s

# pandoc paper.md -o paper.pdf \
# 	--filter pandoc-citeproc \
# 	--bibliography references.bib \
