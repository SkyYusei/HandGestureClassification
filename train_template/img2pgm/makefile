BUILD_DIR = ../../bin/

PGM_WIDTH = 64
PGM_HEIGHT = 36

all: 
	mkdir -p pgm
	ls -l img |tail -n +2 | awk '{print "mkdir -p pgm/"$$9}' | sh
	ls -l img |tail -n +2 | awk '{print "$(BUILD_DIR)/./img2pgm img/"$$9 " "$(PGM_WIDTH)" "$(PGM_HEIGHT)" 0 pgm/" $$9 " "$$9 " 1"}' | sh 

clean:
	rm -r -f pgm


