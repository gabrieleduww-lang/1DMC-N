CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp

build/1dmc-n: src/1dmc-n.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/1dmc-n.cpp -o $@

check: build/1dmc-n
	@python3 tests/check.py

gui: build/1dmc-n
	python3 gui/1dmc-n_gui.py

clean:
	rm -rf build
.PHONY: check gui clean

# Regenerate data/ (needs: pip install numpy periodictable xraydb; see tools/build_data.py header)
data:
	./tools/fetch_and_build.sh
.PHONY: data
