#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {
thread_local const Qwen35UnicodeCategories* g_qwen35_unicode_categories = nullptr;

bool special_type(const GgufTokenizerInfo& m, uint32_t id) {
    if (id >= m.tokens.size()) return false;
    if (m.token_types.size() == m.tokens.size()) {
        const int64_t type = m.token_types[id];
        return type == 3 || type == 4 || type == 5;
    }
    return m.tokens[id].rfind("<|", 0) == 0;
}

size_t char_size(const std::string& s, size_t i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : ((c & 0xe0) == 0xc0 ? 2 : ((c & 0xf0) == 0xe0 ? 3 : 4));
    return std::min(n, s.size() - i);
}

uint32_t codepoint(const std::string& s, size_t i, size_t& n) {
    n = char_size(s, i);
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (n == 1) return c;
    uint32_t v = c & (n == 2 ? 0x1f : (n == 3 ? 0x0f : 0x07));
    for (size_t j = 1; j < n; ++j) v = (v << 6) | (static_cast<unsigned char>(s[i+j]) & 0x3f);
    return v;
}

void append_utf8(uint32_t c, std::string& out) {
    if (c < 0x80) out.push_back(static_cast<char>(c));
    else if (c < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (c >> 6)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3f)));
    } else if (c < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (c >> 12)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (c >> 18)));
        out.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (c & 0x3f)));
    }
}

bool unicode_number(uint32_t c) {
    // Unicode Number categories used by Qwen's \\p{N} pre-tokenizer.
    // Decimal ranges cover the scripts commonly present in model prompts;
    // letter-number and other-number blocks are included explicitly.
    static constexpr uint32_t decimal_starts[] = {
        0x0030,0x0660,0x06f0,0x07c0,0x0966,0x09e6,0x0a66,0x0ae6,0x0b66,
        0x0be6,0x0c66,0x0ce6,0x0d66,0x0de6,0x0e50,0x0ed0,0x0f20,0x1040,
        0x1090,0x17e0,0x1810,0x1946,0x19d0,0x1a80,0x1a90,0x1b50,0x1bb0,
        0x1c40,0x1c50,0xa620,0xa8d0,0xa900,0xa9d0,0xa9f0,0xaa50,0xabf0,
        0xff10,0x104a0,0x10d30,0x11066,0x1d7ce
    };
    for (uint32_t start : decimal_starts) if (c >= start && c < start + 10) return true;
    return (c >= 0x00b2 && c <= 0x00b3) || c == 0x00b9 ||
           (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) ||
           (c >= 0x2160 && c <= 0x2188) || (c >= 0x2460 && c <= 0x249b) ||
           (c >= 0x24ea && c <= 0x24ff) || (c >= 0x2776 && c <= 0x2793) ||
           (c >= 0x3007 && c <= 0x3007) || (c >= 0x3021 && c <= 0x3029) ||
           (c >= 0x3038 && c <= 0x303a) || (c >= 0x3192 && c <= 0x3195) ||
           (c >= 0x3220 && c <= 0x3229) || (c >= 0x3251 && c <= 0x325f) ||
           (c >= 0x3280 && c <= 0x3289) || (c >= 0x32b1 && c <= 0x32bf);
}
bool digit(uint32_t c) { return (c >= '0' && c <= '9') || unicode_number(c); }
bool space(uint32_t c);
bool letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c < 0x80 || space(c) || unicode_number(c)) return false;
    // Exclude punctuation, symbols, arrows, box drawing and emoji blocks before
    // applying script ranges. Combining marks are intentionally included in
    // L|M runs, as required by Qwen's regular expression.
    if ((c >= 0x2000 && c <= 0x2bff) || (c >= 0x3000 && c <= 0x303f) ||
        (c >= 0x1f000 && c <= 0x1ffff) || (c >= 0xfe10 && c <= 0xfe1f) ||
        (c >= 0xfe30 && c <= 0xfe6f) || (c >= 0xff01 && c <= 0xff0f) ||
        (c >= 0xff1a && c <= 0xff20) || (c >= 0xff3b && c <= 0xff40) ||
        (c >= 0xff5b && c <= 0xff65)) return false;
    if (c >= 0x0600 && c <= 0x06ff &&
        (c <= 0x0605 || c == 0x0609 || c == 0x060a || c == 0x060c ||
         c == 0x060d || c == 0x061b || c == 0x061e || c == 0x061f ||
         (c >= 0x066a && c <= 0x066d) || c == 0x06d4)) return false;
    return (c >= 0x00c0 && c <= 0x02ff) ||
           (c >= 0x0300 && c <= 0x036f) ||
           (c >= 0x0370 && c <= 0x052f) ||
           (c >= 0x0531 && c <= 0x0588) ||
           (c >= 0x0590 && c <= 0x05ff) ||
           (c >= 0x0600 && c <= 0x08ff) ||
           (c >= 0x0900 && c <= 0x1fff) ||
           (c >= 0x2c00 && c <= 0x2dff) ||
           (c >= 0x3040 && c <= 0x30ff) ||
           (c >= 0x3100 && c <= 0x312f) ||
           (c >= 0x3130 && c <= 0x318f) ||
           (c >= 0x31a0 && c <= 0x31bf) ||
           (c >= 0x3400 && c <= 0x4dbf) ||
           (c >= 0x4e00 && c <= 0x9fff) ||
           (c >= 0xac00 && c <= 0xd7af) ||
           (c >= 0xf900 && c <= 0xfaff) ||
           (c >= 0x10000 && c <= 0x1efff) ||
           (c >= 0x20000 && c <= 0x3134a);
}
bool space(uint32_t c) {
    return c == 0x20 || (c >= 0x09 && c <= 0x0d) || c == 0x85 || c == 0xa0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200a) || c == 0x2028 ||
           c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

bool byte_token(const std::string& s, uint8_t& b) {
    if (s.size() != 6 || s.substr(0, 3) != "<0x" || s[5] != '>') return false;
    auto h=[](char c)->int { if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; };
    int a=h(s[3]), z=h(s[4]); if(a<0||z<0)return false; b=static_cast<uint8_t>((a<<4)|z); return true;
}

class BaseAdapter : public TokenizerAdapter {
protected:
    static std::unordered_map<std::string,uint32_t> token_map(const GgufTokenizerInfo& m) {
        std::unordered_map<std::string,uint32_t> out; out.reserve(m.tokens.size());
        for(uint32_t i=0;i<m.tokens.size();++i) out.emplace(m.tokens[i],i);
        return out;
    }
    static Status validate_ids(const GgufTokenizerInfo& m) {
        if(m.tokens.empty()) return {StatusCode::InvalidArgument,"Tokenizer vocabulary is empty"};
        std::unordered_set<std::string> unique_tokens;
        unique_tokens.reserve(m.tokens.size());
        for (const auto& token : m.tokens) {
            if (token.empty() || !unique_tokens.insert(token).second)
                return {StatusCode::InvalidArgument,"Tokenizer vocabulary contains an empty or duplicate token"};
        }
        if(!m.scores.empty()&&m.scores.size()!=m.tokens.size()) return {StatusCode::InvalidArgument,"Tokenizer scores do not match vocabulary"};
        if(!m.token_types.empty()&&m.token_types.size()!=m.tokens.size()) return {StatusCode::InvalidArgument,"Tokenizer token types do not match vocabulary"};
        const int64_t ids[]={m.bos_token_id,m.eos_token_id,m.unknown_token_id,m.padding_token_id};
        for(int64_t id:ids) if(id!=-1&&(id<0||static_cast<uint64_t>(id)>=m.tokens.size()))
            return {StatusCode::InvalidArgument,"Tokenizer special id is out of range"};
        return Status::success();
    }
    static Status append_piece(const GgufTokenizerInfo& m,
                               const std::unordered_map<std::string,uint32_t>& map,
                               const std::string& piece,std::vector<uint32_t>& out) {
        auto it=map.find(piece); if(it!=map.end()){out.push_back(it->second);return Status::success();}
        if(m.unknown_token_id>=0){out.push_back(static_cast<uint32_t>(m.unknown_token_id));return Status::success();}
        return {StatusCode::Unsupported,"Tokenizer piece has no vocabulary or unknown-token fallback"};
    }
    static void add_bos(const GgufTokenizerInfo& m,std::vector<uint32_t>& out) {
        if(m.add_bos_token&&m.bos_token_id>=0)out.push_back(static_cast<uint32_t>(m.bos_token_id));
    }
    static void add_eos(const GgufTokenizerInfo& m,std::vector<uint32_t>& out) {
        if(m.add_eos_token&&m.eos_token_id>=0)out.push_back(static_cast<uint32_t>(m.eos_token_id));
    }
    static std::vector<std::pair<std::string,uint32_t>> specials(const GgufTokenizerInfo& m) {
        std::vector<std::pair<std::string,uint32_t>> out;
        for(uint32_t i=0;i<m.tokens.size();++i) {
            const bool added = m.token_types.size()==m.tokens.size()
                ? (m.token_types[i]==3 || m.token_types[i]==4)
                : special_type(m,i);
            if(added) out.emplace_back(m.tokens[i],i);
        }
        std::sort(out.begin(),out.end(),[](const auto&a,const auto&b){return a.first.size()>b.first.size();});
        return out;
    }
};

class SentencePieceAdapter final : public BaseAdapter {
public:
    SentencePieceAdapter(std::string model,std::string pre):desc_{std::move(model),std::move(pre),"sentencepiece-score-merging",SupportState::Unverified}{}
    const TokenizerDescriptor& descriptor()const noexcept override{return desc_;}
    Status validate_metadata(const GgufTokenizerInfo&m)const override{
        Status s=validate_ids(m); if(!s.ok())return s;
        if(m.scores.empty())return {StatusCode::Unsupported,"SentencePiece scores are required"};
        for (float score : m.scores) {
            if (!std::isfinite(score))
                return {StatusCode::InvalidArgument,"SentencePiece scores must all be finite"};
        }
        return Status::success();
    }
    Status encode(const GgufTokenizerInfo&m,const std::string& text,std::vector<uint32_t>& out)const override{
        out.clear(); Status s=validate_metadata(m); if(!s.ok())return s;
        const auto map=token_map(m); const auto special=specials(m); add_bos(m,out);
        size_t pos=0;
        while(pos<text.size()){
            size_t found=std::string::npos,len=0;uint32_t id=0;
            for(const auto& item:special){size_t p=text.find(item.first,pos);if(p<found||(p==found&&item.first.size()>len)){found=p;len=item.first.size();id=item.second;}}
            size_t end=found==std::string::npos?text.size():found;
            if(end>pos){s=encode_plain(m,map,text.substr(pos,end-pos),out);if(!s.ok()){out.clear();return s;}}
            if (found == std::string::npos) break;
            out.push_back(id);
            pos = found + len;
        }
        add_eos(m,out);return Status::success();
    }
    Status decode(const GgufTokenizerInfo&m,const std::vector<uint32_t>&ids,std::string&out)const override{
        out.clear();
        for(uint32_t id:ids){if(id>=m.tokens.size())return {StatusCode::InvalidArgument,"SentencePiece token id out of range"};if(special_type(m,id))continue;
            uint8_t b=0;if(byte_token(m.tokens[id],b)){out.push_back(static_cast<char>(b));continue;}
            std::string piece=m.tokens[id];size_t p=0;while((p=piece.find("\xe2\x96\x81",p))!=std::string::npos){piece.replace(p,3," ");++p;}out+=piece;}
        if (m.add_space_prefix && !out.empty() && out[0] == ' ') out.erase(out.begin());
        return Status::success();
    }
private:
    Status encode_plain(const GgufTokenizerInfo&m,const std::unordered_map<std::string,uint32_t>&map,
                        const std::string&input,std::vector<uint32_t>&out)const{
        // SentencePiece's Unigram model is a maximum-score segmentation problem.
        // Greedy adjacent-pair merging is BPE behavior and can select a lower
        // total-score path, so use Viterbi over UTF-8 byte offsets instead.
        std::string norm;bool pending=false;
        for(size_t input_pos=0;input_pos<input.size();){
            size_t width=0;
            const uint32_t cp=codepoint(input,input_pos,width);
            if(m.remove_extra_whitespaces&&space(cp)){pending=!norm.empty();input_pos+=width;continue;}
            if (pending) {
                if (m.escape_whitespaces) norm += "\xe2\x96\x81";
                else norm.push_back(' ');
                pending = false;
            }
            if(m.escape_whitespaces&&cp==' ')norm+="\xe2\x96\x81";
            else norm.append(input,input_pos,width);
            input_pos+=width;
        }
        if(m.add_space_prefix&&!norm.empty()&&norm.rfind("\xe2\x96\x81",0)!=0)norm.insert(0,"\xe2\x96\x81");
        if (norm.empty()) return Status::success();

        size_t max_piece_bytes = 1;
        for (size_t token_id = 0; token_id < m.tokens.size(); ++token_id) {
            const auto& token = m.tokens[token_id];
            if (!token.empty() && !special_type(m, static_cast<uint32_t>(token_id)))
                max_piece_bytes = std::max(max_piece_bytes, token.size());
        }
        const size_t n = norm.size();
        const double unreachable = -std::numeric_limits<double>::infinity();
        std::vector<double> best(n + 1, unreachable);
        std::vector<size_t> previous(n + 1, n + 1);
        std::vector<uint32_t> previous_token(n + 1, UINT32_MAX);
        best[0] = 0.0;

        auto relax = [&](size_t from, size_t to, uint32_t token_id) {
            if (token_id >= m.scores.size() || !std::isfinite(m.scores[token_id])) return;
            const double score = best[from] + static_cast<double>(m.scores[token_id]);
            if (score > best[to]) {
                best[to] = score;
                previous[to] = from;
                previous_token[to] = token_id;
            }
        };

        static constexpr char hex[] = "0123456789ABCDEF";
        for (size_t start = 0; start < n; ++start) {
            if (!std::isfinite(best[start])) continue;
            const size_t limit = start + std::min(max_piece_bytes, n - start);
            for (size_t end = start + 1; end <= limit; ++end) {
                const auto found = map.find(norm.substr(start, end - start));
                if (found != map.end() && !special_type(m, found->second))
                    relax(start, end, found->second);
            }

            // SentencePiece byte fallback uses explicit <0xHH> vocabulary pieces.
            const unsigned char byte = static_cast<unsigned char>(norm[start]);
            std::string byte_piece = "<0x00>";
            byte_piece[3] = hex[byte >> 4];
            byte_piece[4] = hex[byte & 15];
            const auto byte_found = map.find(byte_piece);
            if (byte_found != map.end() && !special_type(m, byte_found->second))
                relax(start, start + 1, byte_found->second);
            else if (m.unknown_token_id >= 0) {
                const size_t width = char_size(norm, start);
                relax(start, std::min(n, start + width),
                      static_cast<uint32_t>(m.unknown_token_id));
            }
        }

        if (!std::isfinite(best[n]))
            return {StatusCode::Unsupported,"SentencePiece text cannot be segmented by the vocabulary"};
        std::vector<uint32_t> reversed;
        for (size_t position = n; position > 0;) {
            if (previous[position] >= position || previous_token[position] == UINT32_MAX)
                return {StatusCode::BackendFailure,"SentencePiece Viterbi backtrace is inconsistent"};
            reversed.push_back(previous_token[position]);
            position = previous[position];
        }
        out.insert(out.end(), reversed.rbegin(), reversed.rend());
        return Status::success();
    }
    TokenizerDescriptor desc_;
};

class BpeAdapter final : public BaseAdapter {
public:
    explicit BpeAdapter(std::string pre):desc_{"gpt2",pre,"byte-level-bpe",
        // Qwen3.5 and Qwen2/Qwen3 byte-level BPE profiles are checked against
        // pinned upstream tokenizer.json assets in CI.
        (pre == "qwen35" || pre == "qwen2") ? SupportState::Validated : SupportState::Unverified}{build_bytes();}
    const TokenizerDescriptor& descriptor()const noexcept override{return desc_;}
    Status validate_metadata(const GgufTokenizerInfo&m)const override{
        Status s=validate_ids(m);if(!s.ok())return s;
        if(m.merges.empty())return {StatusCode::Unsupported,"BPE merge table is empty"};
        if (desc_.pre_tokenizer == "qwen35" || desc_.pre_tokenizer == "qwen2") {
            // The tested Qwen3.5 and Qwen2/Qwen3 byte-level BPE profiles require
            // complete token types and no implicit BOS/EOS, prefix-space or byte fallback.
            if (m.token_types.size() != m.tokens.size() ||
                m.add_bos_token || m.add_eos_token ||
                m.add_space_prefix || m.byte_fallback) {
                return {StatusCode::Unsupported,"Qwen byte-level BPE requires a complete token-type table without implicit BOS/EOS, prefix-space or byte-fallback behavior"};
            }
        }
        const auto vocabulary=token_map(m);
        if (desc_.pre_tokenizer == "qwen35" || desc_.pre_tokenizer == "qwen2") {
            for (uint32_t byte = 0; byte < 256; ++byte) {
                const auto symbol = encoder_.find(static_cast<uint8_t>(byte));
                if (symbol == encoder_.end() || vocabulary.find(symbol->second) == vocabulary.end())
                    return {StatusCode::Unsupported,"Qwen byte-level vocabulary is missing one of the 256 base byte symbols"};
            }
        }
        std::unordered_set<std::string> unique_merges;
        unique_merges.reserve(m.merges.size());
        for(const auto& merge:m.merges){
            const size_t separator=merge.find(' ');
            if(separator==std::string::npos||separator==0||separator+1>=merge.size()||
               merge.find(' ',separator+1)!=std::string::npos)
                return {StatusCode::InvalidArgument,"Malformed BPE merge entry"};
            const std::string key=merge.substr(0,separator)+"\x1f"+merge.substr(separator+1);
            if(!unique_merges.insert(key).second)
                return {StatusCode::InvalidArgument,"BPE merge table contains a duplicate pair"};
            if(vocabulary.find(merge.substr(0,separator))==vocabulary.end() ||
               vocabulary.find(merge.substr(separator+1))==vocabulary.end() ||
               vocabulary.find(merge.substr(0,separator)+merge.substr(separator+1))==vocabulary.end())
                return {StatusCode::InvalidArgument,"BPE merge input or result is missing from the token vocabulary"};
        }
        return Status::success();
    }
    Status encode(const GgufTokenizerInfo&m,const std::string&text,std::vector<uint32_t>&out)const override{
        out.clear();Status s=validate_metadata(m);if(!s.ok())return s;const auto map=token_map(m);std::unordered_map<std::string,uint32_t> ranks;
        for(uint32_t i=0;i<m.merges.size();++i){size_t p=m.merges[i].find(' ');ranks.emplace(m.merges[i].substr(0,p)+"\x1f"+m.merges[i].substr(p+1),i);}
        const auto special=specials(m);add_bos(m,out);size_t pos=0;
        while(pos<text.size()){size_t found=std::string::npos,len=0;uint32_t id=0;
            for(const auto&item:special){size_t p=text.find(item.first,pos);if(p<found||(p==found&&item.first.size()>len)){found=p;len=item.first.size();id=item.second;}}
            size_t end=found==std::string::npos?text.size():found;if(end>pos){for(const auto&chunk:split(text.substr(pos,end-pos))){s=encode_chunk(m,map,ranks,chunk,out);if(!s.ok()){out.clear();return s;}}}
            if (found == std::string::npos) break;
            out.push_back(id);
            pos = found + len;
        }
        add_eos(m,out);return Status::success();
    }
    Status decode(const GgufTokenizerInfo&m,const std::vector<uint32_t>&ids,std::string&out)const override{
        out.clear();
        for(uint32_t id:ids){
            if(id>=m.tokens.size())return {StatusCode::InvalidArgument,"BPE token id out of range"};
            if(m.token_types.size()==m.tokens.size()){
                const int64_t type=m.token_types[id];
                if(type==3||type==5)continue; // CONTROL and UNUSED
                if(type==4){out+=m.tokens[id];continue;} // USER_DEFINED is isolated but not a skippable special token
            }else if(special_type(m,id))continue;
            uint8_t b=0;
            if(byte_token(m.tokens[id],b)){out.push_back(static_cast<char>(b));continue;}
            const auto&piece=m.tokens[id];
            for(size_t i=0;i<piece.size();){
                size_t n=0;uint32_t cp=codepoint(piece,i,n);
                auto it=decoder_.find(cp);
                if(it==decoder_.end())out.append(piece,i,n);else out.push_back(static_cast<char>(it->second));
                i+=n;
            }
        }
        return Status::success();
    }
private:
    enum class QwenKind : uint8_t { LetterMark, Number, Space, Newline, Other };
    static QwenKind qwen_kind(uint32_t cp) {
        if (g_qwen35_unicode_categories) {
            const auto found = g_qwen35_unicode_categories->find(cp);
            if (found != g_qwen35_unicode_categories->end() && found->second <= 4)
                return static_cast<QwenKind>(found->second);
        }
        if (cp == '\r' || cp == '\n') return QwenKind::Newline;
        if (unicode_number(cp)) return QwenKind::Number;
        if (letter(cp)) return QwenKind::LetterMark;
        if (space(cp)) return QwenKind::Space;
        return QwenKind::Other;
    }
    static size_t contraction_length(const std::string& s, size_t at) {
        static constexpr const char* suffixes[] = {"'s","'t","'re","'ve","'m","'ll","'d"};
        if (at >= s.size() || s[at] != '\'') return 0;
        for (const char* suffix : suffixes) {
            const size_t length = std::char_traits<char>::length(suffix);
            if (at + length > s.size()) continue;
            bool match = true;
            for (size_t i = 0; i < length; ++i) {
                const auto actual = static_cast<unsigned char>(s[at + i]);
                const auto expected = static_cast<unsigned char>(suffix[i]);
                if (std::tolower(actual) != std::tolower(expected)) { match = false; break; }
            }
            if (match) return length;
        }
        return 0;
    }
    static std::vector<std::string> split_qwen35(const std::string& s) {
        struct Character { size_t offset; uint32_t cp; QwenKind kind; };
        std::vector<Character> chars;
        for (size_t i = 0; i < s.size();) {
            size_t width = 0;
            const uint32_t cp = codepoint(s, i, width);
            if (width == 0) break;
            chars.push_back({i, cp, qwen_kind(cp)});
            i += width;
        }
        const size_t count = chars.size();
        auto kind_at = [&](size_t i) { return i < count ? chars[i].kind : QwenKind::Other; };
        auto is_lm = [](QwenKind k) { return k == QwenKind::LetterMark; };
        auto is_space = [](QwenKind k) { return k == QwenKind::Space || k == QwenKind::Newline; };
        auto run = [&](size_t i, const auto& predicate) {
            while (i < count && predicate(chars[i].kind)) ++i;
            return i;
        };
        auto byte_at = [&](size_t i) { return i < count ? chars[i].offset : s.size(); };
        std::vector<std::string> out;
        size_t i = 0;
        while (i < count) {
            const uint32_t cp = chars[i].cp;
            const QwenKind current = chars[i].kind;
            const QwenKind next = kind_at(i + 1);
            size_t end = i;
            const size_t contraction = contraction_length(s, chars[i].offset);
            if (contraction != 0) {
                size_t target = chars[i].offset + contraction;
                end = i;
                while (end < count && chars[end].offset < target) ++end;
            } else if (current != QwenKind::Newline && current != QwenKind::LetterMark &&
                       current != QwenKind::Number && i + 1 < count && is_lm(next)) {
                end = run(i + 1, is_lm);
            } else if (is_lm(current)) {
                end = run(i, is_lm);
            } else if (current == QwenKind::Number) {
                end = i + 1;
            } else if ((cp == ' ' && next == QwenKind::Other) || current == QwenKind::Other) {
                const size_t start = current == QwenKind::Other ? i : i + 1;
                end = run(start, [](QwenKind k) { return k == QwenKind::Other; });
                end = run(end, [](QwenKind k) { return k == QwenKind::Newline; });
            } else {
                const size_t whitespace_end = run(i, is_space);
                size_t last_newline = count;
                for (size_t j = whitespace_end; j > i; --j) {
                    if (chars[j - 1].kind == QwenKind::Newline) { last_newline = j - 1; break; }
                }
                if (last_newline != count) end = last_newline + 1;
                else if (whitespace_end < count && whitespace_end - i >= 2) end = whitespace_end - 1;
                else end = whitespace_end;
            }
            if (end <= i) end = i + 1;
            out.emplace_back(s.substr(chars[i].offset, byte_at(end) - chars[i].offset));
            i = end;
        }
        return out;
    }
    void build_bytes(){
        std::vector<uint32_t> bytes, cps;for(uint32_t b=33;b<=126;++b)bytes.push_back(b);for(uint32_t b=161;b<=172;++b)bytes.push_back(b);for(uint32_t b=174;b<=255;++b)bytes.push_back(b);cps=bytes;
        uint32_t extra=0;for(uint32_t b=0;b<256;++b)if(std::find(bytes.begin(),bytes.end(),b)==bytes.end()){bytes.push_back(b);cps.push_back(256+extra++);}
        for(size_t i=0;i<bytes.size();++i){std::string enc;append_utf8(cps[i],enc);encoder_[static_cast<uint8_t>(bytes[i])]=enc;decoder_[cps[i]]=static_cast<uint8_t>(bytes[i]);}
    }
    std::vector<std::string> split(const std::string&s)const{
        if (desc_.pre_tokenizer == "qwen35") return split_qwen35(s);
        std::vector<std::string> out;size_t i=0;const size_t digit_group=(desc_.pre_tokenizer=="qwen2"||desc_.pre_tokenizer=="qwen3")?1:3;
        while(i<s.size()){size_t n=0;uint32_t cp=codepoint(s,i,n);size_t start=i;
            // GPT-2/Qwen-style pre-tokenizers keep common English contractions
            // as independent pieces before byte-level BPE merge ranking.
            if (s[i] == '\'') {
                static const char* contractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
                size_t contraction_length = 0;
                for (const char* suffix : contractions) {
                    const size_t length = std::char_traits<char>::length(suffix);
                    if (i + length > s.size()) continue;
                    bool matches = true;
                    for (size_t j = 0; j < length; ++j) {
                        const unsigned char actual = static_cast<unsigned char>(s[i + j]);
                        const unsigned char expected = static_cast<unsigned char>(suffix[j]);
                        if (std::tolower(actual) != std::tolower(expected)) { matches = false; break; }
                    }
                    if (matches) { contraction_length = length; break; }
                }
                if (contraction_length != 0) {
                    out.emplace_back(s.substr(i, contraction_length));
                    i += contraction_length;
                    continue;
                }
            }
            if (!space(cp) && !letter(cp) && !digit(cp) && i + n < s.size()) {
                size_t next_n = 0;
                const uint32_t next = codepoint(s, i + n, next_n);
                if (letter(next)) {
                    i += n + next_n;
                    while (i < s.size()) {
                        size_t k = 0;
                        const uint32_t current = codepoint(s, i, k);
                        if (!letter(current)) break;
                        i += k;
                    }
                    out.emplace_back(s.substr(start, i - start));
                    continue;
                }
            }
            if(cp==0x20 && i+n<s.size()){
                size_t next_n=0;uint32_t next=codepoint(s,i+n,next_n);
                if(!space(next)){
                    i+=n;
                    if(letter(next)){i+=next_n;while(i<s.size()){size_t k=0;uint32_t c=codepoint(s,i,k);if(!letter(c))break;i+=k;}}
                    else if(digit(next)){size_t count=0;while(i<s.size()&&count<digit_group){size_t k=0;uint32_t c=codepoint(s,i,k);if(!digit(c))break;i+=k;++count;}}
                    else{i+=next_n;while(i<s.size()){size_t k=0;uint32_t c=codepoint(s,i,k);if(space(c)||letter(c)||digit(c))break;i+=k;}while(i<s.size()&&(s[i]=='\r'||s[i]=='\n'))++i;}
                    out.emplace_back(s.substr(start,i-start));continue;
                }
            }
            if(space(cp)){
                size_t last_space_start=i;
                while(i<s.size()){
                    size_t k=0;uint32_t c=codepoint(s,i,k);if(!space(c))break;
                    last_space_start=i;i+=k;
                }
                // Qwen2's GPT-style regex leaves the final ASCII space for the
                // optional-space prefix of the following word/punctuation run.
                // Earlier spaces form a separate whitespace token.
                if(i<s.size()&&i-last_space_start>=1&&s[last_space_start]==' '&&
                   last_space_start>start) i=last_space_start;
            }
            else if(letter(cp)){i+=n;while(i<s.size()){size_t k=0;uint32_t c=codepoint(s,i,k);if(!letter(c))break;i+=k;}}
            else if(digit(cp)){size_t count=0;while(i<s.size()&&count<digit_group){size_t k=0;uint32_t c=codepoint(s,i,k);if(!digit(c))break;i+=k;++count;}}
            else{i+=n;while(i<s.size()){size_t k=0;uint32_t c=codepoint(s,i,k);if(space(c)||letter(c)||digit(c))break;i+=k;}while(i<s.size()&&(s[i]=='\r'||s[i]=='\n'))++i;}
            out.emplace_back(s.substr(start,i-start));
        }return out;
    }
    Status encode_chunk(const GgufTokenizerInfo&m,const std::unordered_map<std::string,uint32_t>&map,
                        const std::unordered_map<std::string,uint32_t>&ranks,const std::string&chunk,std::vector<uint32_t>&out)const{
        std::vector<std::string> symbols;for(unsigned char b:chunk)symbols.push_back(encoder_.at(b));
        while(symbols.size()>1){uint32_t best=UINT32_MAX;size_t at=symbols.size();for(size_t i=0;i+1<symbols.size();++i){auto it=ranks.find(symbols[i]+"\x1f"+symbols[i+1]);if(it!=ranks.end()&&it->second<best){best=it->second;at=i;}}
            if (at == symbols.size()) break;
            symbols[at] += symbols[at + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(at + 1));
        }
        for(const auto&p:symbols){Status s=append_piece(m,map,p,out);if(!s.ok())return s;}return Status::success();
    }
    TokenizerDescriptor desc_;std::unordered_map<uint8_t,std::string> encoder_;std::unordered_map<uint32_t,uint8_t> decoder_;
};

} // namespace

void set_qwen35_unicode_categories(const Qwen35UnicodeCategories* categories) noexcept {
    g_qwen35_unicode_categories = categories;
}

Status register_builtin_tokenizer_adapters(TokenizerRegistry& registry) {
    Status s=registry.register_adapter(std::make_shared<SentencePieceAdapter>("llama","default"));if(!s.ok())return s;
    s=registry.register_adapter(std::make_shared<SentencePieceAdapter>("llama","llama"));if(!s.ok())return s;
    s=registry.register_adapter(std::make_shared<SentencePieceAdapter>("spm","default"));if(!s.ok())return s;
    const char* profiles[]={"gpt2","llama3","qwen2","qwen3","qwen35"};
    for(const char* p:profiles){s=registry.register_adapter(std::make_shared<BpeAdapter>(p));if(!s.ok())return s;}
    return Status::success();
}

} // namespace ai_chat::gpu_engine
