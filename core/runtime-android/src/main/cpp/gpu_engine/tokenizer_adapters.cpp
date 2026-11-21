#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {

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

bool letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c < 0x80 || std::isspace(static_cast<unsigned char>(c)) || (c >= '0' && c <= '9')) return false;
    if ((c >= 0x2000 && c <= 0x206f) || (c >= 0x3000 && c <= 0x303f)) return false;
    return c >= 0x00c0 && c <= 0x1efff;
}
bool digit(uint32_t c) { return (c >= '0' && c <= '9') || (c >= 0x0660 && c <= 0x0669); }
bool space(uint32_t c) { return c == 0x20 || (c >= 0x09 && c <= 0x0d) || c == 0xa0 || c == 0x3000; }

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
        for(uint32_t i=0;i<m.tokens.size();++i) if(special_type(m,i)) out.emplace_back(m.tokens[i],i);
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
            if(pending){norm.push_back(' ');pending=false;}
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
    explicit BpeAdapter(std::string pre):desc_{"gpt2",std::move(pre),"byte-level-bpe",SupportState::Unverified}{build_bytes();}
    const TokenizerDescriptor& descriptor()const noexcept override{return desc_;}
    Status validate_metadata(const GgufTokenizerInfo&m)const override{
        Status s=validate_ids(m);if(!s.ok())return s;if(m.merges.empty())return {StatusCode::Unsupported,"BPE merge table is empty"};
        for(const auto&x:m.merges)if(x.find(' ')==std::string::npos)return {StatusCode::InvalidArgument,"Malformed BPE merge entry"};
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
        out.clear();for(uint32_t id:ids){if(id>=m.tokens.size())return {StatusCode::InvalidArgument,"BPE token id out of range"};if(special_type(m,id))continue;
            uint8_t b=0;if(byte_token(m.tokens[id],b)){out.push_back(static_cast<char>(b));continue;}
            const auto&piece=m.tokens[id];for(size_t i=0;i<piece.size();){size_t n=0;uint32_t cp=codepoint(piece,i,n);auto it=decoder_.find(cp);if(it==decoder_.end())out.append(piece,i,n);else out.push_back(static_cast<char>(it->second));i+=n;}}
        return Status::success();
    }
private:
    void build_bytes(){
        std::vector<uint32_t> bytes, cps;for(uint32_t b=33;b<=126;++b)bytes.push_back(b);for(uint32_t b=161;b<=172;++b)bytes.push_back(b);for(uint32_t b=174;b<=255;++b)bytes.push_back(b);cps=bytes;
        uint32_t extra=0;for(uint32_t b=0;b<256;++b)if(std::find(bytes.begin(),bytes.end(),b)==bytes.end()){bytes.push_back(b);cps.push_back(256+extra++);}
        for(size_t i=0;i<bytes.size();++i){std::string enc;append_utf8(cps[i],enc);encoder_[static_cast<uint8_t>(bytes[i])]=enc;decoder_[cps[i]]=static_cast<uint8_t>(bytes[i]);}
    }
    std::vector<std::string> split(const std::string&s)const{
        std::vector<std::string> out;size_t i=0;const size_t digit_group=(desc_.pre_tokenizer=="qwen2"||desc_.pre_tokenizer=="qwen3")?1:3;
        while(i<s.size()){size_t n=0;uint32_t cp=codepoint(s,i,n);size_t start=i;
            // GPT-2/Qwen-style pre-tokenizers keep common English contractions
            // as independent pieces before byte-level BPE merge ranking.
            if (s[i] == '\\'') {
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
            if(space(cp)){i+=n;while(i<s.size()){size_t k=0;uint32_t c=codepoint(s,i,k);if(!space(c))break;i+=k;}}
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

Status register_builtin_tokenizer_adapters(TokenizerRegistry& registry) {
    Status s=registry.register_adapter(std::make_shared<SentencePieceAdapter>("llama","default"));if(!s.ok())return s;
    s=registry.register_adapter(std::make_shared<SentencePieceAdapter>("llama","llama"));if(!s.ok())return s;
    s=registry.register_adapter(std::make_shared<SentencePieceAdapter>("spm","default"));if(!s.ok())return s;
    const char* profiles[]={"gpt2","llama3","qwen2","qwen3"};
    for(const char* p:profiles){s=registry.register_adapter(std::make_shared<BpeAdapter>(p));if(!s.ok())return s;}
    return Status::success();
}

} // namespace ai_chat::gpu_engine
