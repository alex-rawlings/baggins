# needed for latexmk to handle the dependencies
.PHONY: paper.pdf

all: paper.pdf

paper.pdf : paper.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode -file-line-error" paper.tex

#delete all temporary files and the output .pdf file
clean :
	latexmk -C
