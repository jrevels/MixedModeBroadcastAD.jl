SRCS = arity compute
PDFS = $(SRCS:=.pdf)

.PHONY: all
all: $(PDFS)

.PHONY: clean
clean:
	$(RM) $(PDFS)

%.pdf: %.tex common.tex *.csv
	rubber --pdf $<
