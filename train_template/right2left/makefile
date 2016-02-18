BUILD_DIR = ../../bin/

PGM_WIDTH = 64
PGM_HEIGHT = 36
TYPE = jpg
all: 
	mkdir -p pgm
	ls -l img | tail -n +2 | awk '{print "mkdir -p pgm/"$$9}' | sh 
	ls -l img | tail -n +2| awk '{print "$(BUILD_DIR)/./img2pgm img/"$$9" $(PGM_WIDTH) $(PGM_HEIGHT) 0 pgm/"$$9" "$$9" 1"}' | sh
	find ./pgm/ -name '*.pgm' | awk '{print "convert " $$0 " -flop " $$0}' | sh

clean:
	rm -r -f pgm


