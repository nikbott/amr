CXX      = mpicxx
CXXFLAGS = -std=c++20 -O3 -fopenmp -march=native -Wall -Wextra -Wpedantic -Wshadow -Wformat=2
LDFLAGS  = 
TESTLIBS = -lCatch2Main -lCatch2

TARGET   = amr
TESTS    = test

HEADERS  = core.hpp tree.hpp physics.hpp viz.hpp

all: $(TARGET) $(TESTS)

$(TARGET): main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET) $(LDFLAGS)

$(TESTS): tests.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) tests.cpp -o $(TESTS) $(LDFLAGS) $(TESTLIBS)

clean:
	rm -f $(TARGET) $(TESTS) *.vtu *.pvtu