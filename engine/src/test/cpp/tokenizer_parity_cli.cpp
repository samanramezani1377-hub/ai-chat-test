#include "gguf_reader.h"
#include "tokenizer.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace ai_chat::gpu_engine;
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: tokenizer_parity_cli MODEL.gguf TEXT CATEGORIES.txt\n"; return 2; }
    GgufReader reader;
    Status status = GgufReader::open(argv[1], reader);
    if (!status.ok()) { std::cerr << "GGUF read failed: " << status.message << "\n"; return 3; }
    TokenizerRegistry registry;
    status = register_builtin_tokenizer_adapters(registry);
    if (!status.ok()) { std::cerr << "Tokenizer registry setup failed: " << status.message << "\n"; return 4; }
    const auto adapter = registry.find(reader.info().tokenizer);
    if (!adapter) { std::cerr << "No tokenizer adapter matches GGUF metadata\n"; return 5; }
    status = adapter->validate_metadata(reader.info().tokenizer);
    if (!status.ok()) { std::cerr << "Tokenizer metadata rejected: " << status.message << "\n"; return 6; }
    Qwen35UnicodeCategories categories;
    std::ifstream category_input(argv[3]);
    if (!category_input) { std::cerr << "Cannot read Unicode category fixture\n"; return 7; }
    uint32_t codepoint = 0; unsigned int kind = 0;
    while (category_input >> codepoint >> kind) {
        if (kind > 4) { std::cerr << "Invalid Unicode category value\n"; return 8; }
        categories[codepoint] = static_cast<uint8_t>(kind);
    }
    set_qwen35_unicode_categories(&categories);
    std::vector<uint32_t> ids;
    status = adapter->encode(reader.info().tokenizer, argv[2], ids);
    if (!status.ok()) { set_qwen35_unicode_categories(nullptr); std::cerr << "Tokenizer encode failed: " << status.message << "\n"; return 9; }
    std::string decoded;
    status = adapter->decode(reader.info().tokenizer, ids, decoded);
    set_qwen35_unicode_categories(nullptr);
    if (!status.ok()) { std::cerr << "Tokenizer decode failed: " << status.message << "\n"; return 10; }
    for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ","; std::cout << ids[i]; }
    std::cout << "\n";
    static constexpr char hex[] = "0123456789abcdef";
    for (unsigned char byte : decoded) std::cout << hex[byte >> 4] << hex[byte & 0x0f];
    std::cout << "\n";
    return 0;
}
