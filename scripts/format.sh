#!/usr/bin/env bash
# Formats NetBender C++ source files using clang-format
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v clang-format &> /dev/null; then
    CLANG_FORMAT_BIN="$(command -v clang-format)"
elif [ -x /opt/homebrew/bin/clang-format ]; then
    CLANG_FORMAT_BIN="/opt/homebrew/bin/clang-format"
else
    echo "Error: clang-format is not installed or not in PATH."
    echo "Install via Homebrew: brew install clang-format"
    exit 1
fi

echo "Running clang-format on src/backend (dsp, engine, bending), src/shared, src/tests, and src/frontend/juce..."

find "$ROOT_DIR/src/backend" "$ROOT_DIR/src/shared" "$ROOT_DIR/src/tests" \
    -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) \
    -exec "$CLANG_FORMAT_BIN" -i -style=file:"$ROOT_DIR/.clang-format" {} +

# Run on custom frontend code, strictly excluding vendor/framework trees
find "$ROOT_DIR/src/frontend/juce" \
    -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) \
    ! -path "*/JuceLibraryCode/*" \
    ! -path "*/juce_source/*" \
    -exec "$CLANG_FORMAT_BIN" -i -style=file:"$ROOT_DIR/.clang-format" {} +

echo "Clang formatting complete!"
