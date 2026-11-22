Sources TikZ des figures du README. Pour regénérer une image (pdflatex + poppler) :

    pdflatex architecture.tex && pdftoppm -png -r 220 -singlefile architecture.pdf ../images/architecture
    pdflatex pipeline.tex     && pdftoppm -png -r 220 -singlefile pipeline.pdf     ../images/pipeline
    pdflatex resultats.tex    && pdftoppm -png -r 200 -singlefile resultats.pdf    ../images/resultats
