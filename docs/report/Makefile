TARGET=main
TEX_SOURCES=$(TARGET).tex *.tex

all: fast

fast: $(TEX_SOURCES)
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)

biber: clean
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)
	biber $(TARGET)
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)

index:
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)
	makeindex $(TARGET)
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)
	pdflatex -shell-escape -interaction=nonstopmode $(TARGET)

clean:
	rm -f *.log *.toc *.lof *.lot *.out *.pdf *.aux *.bbl *.blg *.idx *.ilg *.ind *.lol *.bcf *.run.xml
	rm -rf _minted-$(TARGET)

.PHONY: biber index

