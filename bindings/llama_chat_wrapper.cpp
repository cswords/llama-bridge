/**
 * Llama Chat Bindings
 *
 * pybind11 wrapper for llama.cpp's chat functionality.
 * Exposes chat template handling and inference to Python.
 */

#include <iostream>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

namespace py = pybind11;

/**
 * Wrapper class for llama.cpp chat functionality.
 */
class LlamaChatWrapper {
public:
  LlamaChatWrapper(const std::string &model_path, int32_t n_ctx,
                   int32_t n_batch, int32_t n_ubatch, int32_t n_threads,
                   bool flash_attn)
      : model_path_(model_path) {
    // Initialize llama backend
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);

    if (!model_) {
      llama_backend_free();
      throw std::runtime_error("Failed to load model: " + model_path);
    }

    vocab_ = llama_model_get_vocab(model_);

    // Initialize chat templates from model
    templates_ = common_chat_templates_init(model_, "", "", "");

    if (!templates_) {
      llama_model_free(model_);
      llama_backend_free();
      throw std::runtime_error("Failed to initialize chat templates");
    }

    // Resolve context size
    if (n_ctx <= 0) {
      n_ctx = llama_model_n_ctx_train(model_);
      // Cap at 128k by default
      if (n_ctx > 131072)
        n_ctx = 131072;
    }

    // Resolve batch size
    if (n_batch <= 0) {
      n_batch = std::min((uint32_t)n_ctx, 512u);
    }
    if (n_batch > n_ctx) {
      n_batch = n_ctx;
    }

    // Resolve ubatch
    if (n_ubatch <= 0) {
      n_ubatch = (n_batch > 0) ? n_batch : 512;
    }

    // Initialize context for inference
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.n_threads = (n_threads > 0) ? n_threads : 8;
    ctx_params.n_threads_batch = (n_threads > 0) ? n_threads : 8;

    // Support at least 2 simultaneous sequences (0=main, 1=ephemeral)
    ctx_params.n_seq_max = 2;

    // Apple Silicon optimization hints
    ctx_params.cb_eval = nullptr;

    std::cerr << "DEBUG: Calling llama_init_from_model with n_seq_max="
              << ctx_params.n_seq_max << std::endl;

    ctx_ = llama_init_from_model(model_, ctx_params);

    if (!ctx_) {
      llama_model_free(model_);
      llama_backend_free();
      throw std::runtime_error("Failed to initialize context");
    }

    // Initialize sampler (greedy by default for now)
    auto sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

    loaded_ = true;
  }

  ~LlamaChatWrapper() {
    if (sampler_) {
      llama_sampler_free(sampler_);
    }
    if (ctx_) {
      llama_free(ctx_);
    }
    if (model_) {
      llama_model_free(model_);
    }
    llama_backend_free();
  }

  /**
   * Apply chat template to format messages into a prompt.
   */
  py::dict
  apply_template(const std::vector<std::map<std::string, py::object>> &messages,
                 const std::vector<std::map<std::string, std::string>> &tools,
                 bool add_generation_prompt) {
    common_chat_templates_inputs inputs;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja = true;

    for (const auto &msg : messages) {
      common_chat_msg chat_msg;
      if (msg.count("role"))
        chat_msg.role = msg.at("role").cast<std::string>();
      if (msg.count("content")) {
        auto content = msg.at("content");
        if (py::isinstance<py::str>(content))
          chat_msg.content = content.cast<std::string>();
      }
      if (msg.count("tool_call_id"))
        chat_msg.tool_call_id = msg.at("tool_call_id").cast<std::string>();
      if (msg.count("tool_name"))
        chat_msg.tool_name = msg.at("tool_name").cast<std::string>();

      if (msg.count("tool_calls")) {
        auto tc_obj = msg.at("tool_calls");
        if (py::isinstance<py::list>(tc_obj)) {
          for (auto tc : tc_obj.cast<py::list>()) {
            auto tc_dict = tc.cast<py::dict>();
            common_chat_tool_call tool_call;
            if (tc_dict.contains("name"))
              tool_call.name = tc_dict["name"].cast<std::string>();
            if (tc_dict.contains("arguments"))
              tool_call.arguments = tc_dict["arguments"].cast<std::string>();
            if (tc_dict.contains("id"))
              tool_call.id = tc_dict["id"].cast<std::string>();
            chat_msg.tool_calls.push_back(tool_call);
          }
        }
      }
      inputs.messages.push_back(chat_msg);
    }

    for (const auto &tool : tools) {
      common_chat_tool chat_tool;
      if (tool.count("name"))
        chat_tool.name = tool.at("name");
      if (tool.count("description"))
        chat_tool.description = tool.at("description");
      if (tool.count("parameters"))
        chat_tool.parameters = tool.at("parameters");
      inputs.tools.push_back(chat_tool);
    }

    common_chat_params params =
        common_chat_templates_apply(templates_.get(), inputs);

    py::dict result;
    result["prompt"] = params.prompt;
    result["grammar"] = params.grammar;
    result["format"] = common_chat_format_name(params.format);

    py::list stops;
    for (const auto &stop : params.additional_stops)
      stops.append(stop);
    result["additional_stops"] = stops;

    return result;
  }

  /**
   * Parse model response.
   */
  py::dict parse_response(const std::string &response_text,
                          bool is_partial = false) {
    common_chat_templates_inputs dummy_inputs;
    common_chat_params params =
        common_chat_templates_apply(templates_.get(), dummy_inputs);

    common_chat_syntax syntax;
    syntax.format = params.format;
    if (!params.parser.empty()) {
      syntax.parser.load(params.parser);
    }
    syntax.parse_tool_calls = true;
    syntax.thinking_forced_open = params.thinking_forced_open;

    // Use AUTO reasoning format to let llama.cpp decide best extraction
    syntax.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

    common_chat_msg parsed;
    try {
      parsed = common_chat_parse(response_text, is_partial, syntax);
    } catch (const std::exception &e) {
      // Fallback to raw content if parsing fails
      std::cerr << "WARN: parse_response failed: " << e.what()
                << " - falling back to raw" << std::endl;
      parsed.content = response_text;
    }

    // Hard debug log to stderr (std::cerr)
    // std::cerr << "DEBUG: parse_response: input_len=" <<
    // response_text.length()
    //           << " partial=" << is_partial
    //           << " res_len=" << parsed.reasoning_content.length()
    //           << " cont_len=" << parsed.content.length() << std::endl;

    py::dict result;
    result["content"] = parsed.content;
    result["reasoning_content"] = parsed.reasoning_content;

    py::list tool_calls;
    for (const auto &tc : parsed.tool_calls) {
      py::dict tc_dict;
      tc_dict["name"] = tc.name;
      tc_dict["arguments"] = tc.arguments;
      tc_dict["id"] = tc.id;
      tool_calls.append(tc_dict);
    }
    result["tool_calls"] = tool_calls;

    return result;
  }

  /**
   * Initialize inference with a prompt.
   */
  void init_inference(const std::string &prompt) {
    // tokenize
    std::vector<llama_token> tokens = common_tokenize(ctx_, prompt, true, true);

    // 1. We have a significant history (> 256 tokens)
    // 2. New prompt is significantly shorter (< 50% of history)
    bool is_ephemeral = false;
    if (previous_tokens_.size() > 256 &&
        tokens.size() < (previous_tokens_.size() / 2)) {
      is_ephemeral = true;
    }

    current_seq_id_ = is_ephemeral ? 1 : 0;

    // Ensure we are not exceeding n_seq_max (which we set to 2)
    if (current_seq_id_ >= 2) {
      current_seq_id_ = 0; // Fallback to 0 if out of bounds (should not happen
                           // with our logic)
    }

    // Find common prefix with previously cached tokens
    size_t n_keep = 0;
    if (!is_ephemeral) {
      while (n_keep < tokens.size() && n_keep < previous_tokens_.size() &&
             tokens[n_keep] == previous_tokens_[n_keep]) {
        n_keep++;
      }
    } else {
      // Ephemeral: start from scratch in seq 1
      n_keep = 0;
      llama_memory_t mem = llama_get_memory(ctx_);
      llama_memory_seq_rm(mem, current_seq_id_, -1, -1);
      std::cerr << "DEBUG: Ephemeral request detected (seq_id=1). Not touching "
                   "main cache."
                << std::endl;
    }

    // Invalidate KV cache beyond n_keep
    if (!is_ephemeral) {
      llama_memory_t mem = llama_get_memory(ctx_);

      // 1. Remove tail of main sequence
      bool rm_ok = llama_memory_seq_rm(mem, 0, (llama_pos)n_keep, -1);
      if (!rm_ok) {
        std::cerr << "WARNING: llama_memory_seq_rm(0, " << n_keep
                  << ", -1) returned false!" << std::endl;
        // Try -1 as fallback
        llama_memory_seq_rm(mem, -1, (llama_pos)n_keep, -1);
      }

      // 2. Check overlap logic
      if (n_keep == tokens.size() && n_keep > 0) {
        // Re-decode last token logic...
        n_keep--;
        llama_memory_seq_rm(mem, 0, (llama_pos)n_keep, -1);
        std::cerr
            << "DEBUG: Prompt cache full hit, re-decoding last token at pos "
            << n_keep << std::endl;
      } else {
        std::cerr << "DEBUG: Prompt cache partial hit: n_keep=" << n_keep
                  << " tokens=" << tokens.size() << std::endl;
      }

      // 3. Smart Space Check
      // Calculate how much space we need vs have
      int32_t n_ctx_total = llama_n_ctx(ctx_);

      // Estimate usage using max pos (assuming mostly contiguous or at least
      // highest pos)
      int32_t max0 = (int32_t)llama_memory_seq_pos_max(mem, 0);
      int32_t len0 = (max0 >= 0) ? (max0 + 1) : 0;

      int32_t max1 = (int32_t)llama_memory_seq_pos_max(mem, 1);
      int32_t len1 = (max1 >= 0) ? (max1 + 1) : 0;

      int32_t n_used = len0 + len1;

      int32_t n_needed = (int32_t)tokens.size() - (int32_t)n_keep;

      if (n_used + n_needed > n_ctx_total) {
        std::cerr << "WARN: KV Cache pressure! Used=" << n_used
                  << " + Need=" << n_needed << " > Total=" << n_ctx_total
                  << ". Clearing ephemeral cache (seq 1)." << std::endl;

        // Clear ephemeral sequence completely
        llama_memory_seq_rm(mem, 1, -1, -1);

        // Re-check
        max0 = (int32_t)llama_memory_seq_pos_max(mem, 0);
        len0 = (max0 >= 0) ? (max0 + 1) : 0;
        max1 = (int32_t)llama_memory_seq_pos_max(mem, 1);
        len1 = (max1 >= 0) ? (max1 + 1) : 0;
        n_used = len0 + len1;

        if (n_used + n_needed > n_ctx_total) {
          std::cerr << "CRITICAL: Not enough KV cache space even after cleanup."
                    << std::endl;
          throw std::runtime_error("Context window exceeded (KV Cache Full)");
        }
      }
    }

    // Update usage
    n_prompt_tokens_ = tokens.size();
    n_gen_tokens_ = 0;

    // Decode in batches only the NEW parts
    int32_t n_batch_val = (int32_t)llama_n_batch(ctx_);

    for (size_t i = n_keep; i < tokens.size(); i += n_batch_val) {
      int32_t n_eval = (int32_t)tokens.size() - i;
      if (n_eval > n_batch_val) {
        n_eval = n_batch_val;
      }

      llama_batch batch = llama_batch_init(n_eval, 0, 1);
      for (int32_t j = 0; j < n_eval; j++) {
        common_batch_add(batch, tokens[i + j], (int32_t)(i + j),
                         {current_seq_id_}, false);
      }

      // Only request logits for the last token of the entire prompt
      if (i + n_eval == tokens.size()) {
        batch.logits[batch.n_tokens - 1] = true;
      }

      if (llama_decode(ctx_, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to decode prompt");
      }
      llama_batch_free(batch);
    }

    n_past_ = tokens.size();

    // Only update previous_tokens_ if this is the main sequence
    if (!is_ephemeral) {
      previous_tokens_ = tokens;
    }
  }

  /**
   * Get next token. Returns empty string if EOS.
   */
  std::string get_next_token() {
    // Sample
    llama_token id = llama_sampler_sample(sampler_, ctx_, -1);

    if (llama_vocab_is_eog(vocab_, id)) {
      return "";
    }

    n_gen_tokens_++;

    // Convert to piece
    std::string piece = common_token_to_piece(ctx_, id, true);

    // Decode this token for next turn
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, id, n_past_, {0}, true);

    if (llama_decode(ctx_, batch) != 0) {
      llama_batch_free(batch);
      return ""; // Treat as error/EOS
    }
    llama_batch_free(batch);

    n_past_++;
    previous_tokens_.push_back(id);
    return piece;
  }

  /**
   * Simple non-streaming generate.
   */
  std::string generate(const std::string &prompt, int max_tokens = 4096) {
    init_inference(prompt);
    std::string result;
    for (int i = 0; i < max_tokens; i++) {
      std::string token = get_next_token();
      if (token.empty())
        break;
      result += token;
    }
    return result;
  }

  py::dict get_model_info() const {
    py::dict info;
    info["path"] = model_path_;
    info["loaded"] = loaded_;
    if (model_) {
      info["n_vocab"] = llama_vocab_n_tokens(vocab_);
      info["n_ctx_train"] = llama_model_n_ctx_train(model_);
    }
    return info;
  }

  py::dict get_usage() const {
    py::dict usage;
    usage["prompt_tokens"] = n_prompt_tokens_;
    usage["completion_tokens"] = n_gen_tokens_;
    usage["total_tokens"] = n_prompt_tokens_ + n_gen_tokens_;
    return usage;
  }

private:
  std::string model_path_;
  bool loaded_ = false;
  llama_model *model_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_sampler *sampler_ = nullptr;
  common_chat_templates_ptr templates_;
  std::vector<llama_token> previous_tokens_;
  int current_seq_id_ = 0;
  int n_past_ = 0;
  int n_prompt_tokens_ = 0;
  int n_gen_tokens_ = 0;
};

PYBIND11_MODULE(llama_chat, m) {
  m.doc() = "Llama.cpp chat bindings for Python";
  py::class_<LlamaChatWrapper>(m, "LlamaChatWrapper")
      .def(py::init<const std::string &, int32_t, int32_t, int32_t, int32_t,
                    bool>(),
           py::arg("model_path"), py::arg("n_ctx") = 0, py::arg("n_batch") = 0,
           py::arg("n_ubatch") = 0, py::arg("n_threads") = 0,
           py::arg("flash_attn") = false)
      .def("apply_template", &LlamaChatWrapper::apply_template,
           py::arg("messages"), py::arg("tools"),
           py::arg("add_generation_prompt"))
      .def("parse_response", &LlamaChatWrapper::parse_response,
           py::arg("response_text"), py::arg("is_partial") = false)
      .def("init_inference", &LlamaChatWrapper::init_inference,
           py::arg("prompt"))
      .def("get_next_token", &LlamaChatWrapper::get_next_token)
      .def("generate", &LlamaChatWrapper::generate, py::arg("prompt"),
           py::arg("max_tokens") = 4096)
      .def("get_model_info", &LlamaChatWrapper::get_model_info)
      .def("get_usage", &LlamaChatWrapper::get_usage);
  m.def("version", []() { return "0.1.1"; });
}
