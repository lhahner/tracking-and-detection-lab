# HPS Thesis Template

This LaTeX template has been adapted from the HPS report template and the thesis template of the institute of computer science of the university of Göttingen.

This template provides configurations and examples for creating beatiful and well organized master's and bachelor's theses.

Using the template as it is given is not a requirement for the thesis being accepted and adjustments can be made.
Nevertheless, the layout of the titlepage should be mostly left as it is.

## Usage

The template uses LaTeX, therefore, it must be compiled by a TeX engine supporting pdflatex to produce a PDF file.
This can be a local installation of TeX such as TeX Live or MiKTeX or a web application such as Overleaf or Sharelatex (https://sharelatex.gwdg.de).

To work on this template online, the files tex files as well as the Assets and Figures folders need to be saved as a Zip file and can then be imported via the respective web interface of Overleaf and Sharelatex as a new project.

When working locally with this template, it is recommended to use a LaTeX IDE such as TeXstudio or a programming IDE with a LaTeX plugin.

Furthermore, this template provides a Makefile for producing a PDF from the command line on systems that have make installed.

Use `make` to produce a fast rebuild of the document.
Use `make biber` to make a clean rebuild including the bibliography.

The first build should use `make biber`.

To expand the bibliography, references should be added in the biblatex format to ref.bib.

Usage of svg files is possible but might require adding the flag `--shell-escape` in the Makefile to the pdflatex commands and an installation of Inkscape to enable the conversion of svg images to pdf images as required for adding them into the final PDF.

When starting to work with the template, one should at first configure the document configuration found in main.tex.

## Contact & Credits

This template was adapted from the official thesis template of the institute of computer science by
Lars Quentin (lars.quentin@stud.uni-goettingen.de) and Jonathan Decker (jonathan.decker@uni-goettingen.de).

Please direct any questions regarding the template to them.
