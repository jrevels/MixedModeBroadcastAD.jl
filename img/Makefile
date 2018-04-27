SRCS = arity
PDFS = $(SRCS:=.pdf)

.PHONY: all
all: $(PDFS)

.PHONY: clean
clean:
	$(RM) $(PDFS)

%.pdf: %.tex common.tex
	rubber --pdf $<
