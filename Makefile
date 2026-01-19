.PHONY: build clean test dev

LLAMA_DIR = vendor/llama.cpp
BUILD_DIR = build

# Build llama.cpp and pybind11 bindings
build:
	@echo "Building llama.cpp..."
	cd $(LLAMA_DIR) && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build -j
	
	@echo "Building pybind11 bindings..."
	cd bindings && cmake -B build \
		-DLLAMA_DIR=../$(LLAMA_DIR) \
		-Dpybind11_DIR=$$(python -c "import pybind11; print(pybind11.get_cmake_dir())") \
		&& cmake --build build
	
	@echo "Copying libraries..."
	mkdir -p src/lib
	cp $(LLAMA_DIR)/build/libllama.dylib src/lib/ 2>/dev/null || cp $(LLAMA_DIR)/build/libllama.so src/lib/
	cp $(LLAMA_DIR)/build/libggml*.dylib src/lib/ 2>/dev/null || cp $(LLAMA_DIR)/build/libggml*.so src/lib/ 2>/dev/null || true
	cp bindings/build/llama_chat*.so src/ 2>/dev/null || cp bindings/build/llama_chat*.dylib src/

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(LLAMA_DIR)/build bindings/build
	rm -f src/lib/*.dylib src/lib/*.so src/*.so src/*.dylib

# Run tests
test:
	uv run pytest tests/ -v

# Install dev dependencies and setup environment
dev:
	uv sync --all-extras
	. .venv/bin/activate && nodeenv -p

# Download model helper
hfd:
	@echo "Usage: make hfd REPO=<org/repo> INCLUDE=<pattern>"
	@echo "Example: make hfd REPO=bartowski/Qwen3-Coder-30B-A3B-Instruct-GGUF INCLUDE='*Q5_K_M.gguf'"
ifdef REPO
	bash scripts/hfd.sh $(REPO) --include "$(INCLUDE)" --local-dir models/$(REPO)
endif
