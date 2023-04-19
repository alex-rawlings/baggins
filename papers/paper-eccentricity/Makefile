TARGET = paper-eccentricity

# needed for latexmk to handle the dependencies
.PHONY: $(TARGET).pdf


all: $(TARGET).pdf

$(TARGET).pdf : $(TARGET).tex
	latexmk -pdf -pdflatex="pdflatex -interaction=errorstopmode -file-line-error" $(TARGET).tex

#delete all temporary files and the output .pdf file
clean :
	latexmk -C
