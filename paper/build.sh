pandoc paper.md -o index.html \
	--css style.css \
    --template template.html \
	--mathjax \
	--filter pandoc-fignos \
	--filter pandoc-tablenos \
	--filter pandoc-citeproc \
	--bibliography references.bib \
	-s

# pandoc paper.md -o paper.tex \
# 	--template=template.tex \
# 	--filter pandoc-citeproc \
# 	--bibliography references.bib \

# pandoc paper.tex -o paper.pdf

pandoc paper.md -o paper.pdf \
	--filter pandoc-citeproc \
	--bibliography references.bib \

# pandoc paper.tex -o paper.pdf


# rm paper.tex
